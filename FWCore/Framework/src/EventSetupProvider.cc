// -*- C++ -*-
//
// Package:     Framework
// Module:      EventSetupProvider
//
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:27:14 EST 2005
//

// system include files
#include <algorithm>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/Framework/interface/NumberOfConcurrentIOVs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace eventsetup {

    EventSetupProvider::EventSetupProvider(ActivityRegistry const* activityRegistry,
                                           unsigned subProcessIndex,
                                           const PreferredProviderInfo* iInfo)
        : activityRegistry_(activityRegistry),
          mustFinishConfiguration_(true),
          subProcessIndex_(subProcessIndex),
          preferredProviderInfo_((nullptr != iInfo) ? (new PreferredProviderInfo(*iInfo)) : nullptr),
          finders_(new std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>()),
          dataProviders_(new std::vector<std::shared_ptr<DataProxyProvider>>()),
          referencedDataKeys_(new std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription const*>>),
          recordToFinders_(
              new std::map<EventSetupRecordKey, std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>>),
          psetIDToRecordKey_(new std::map<ParameterSetIDHolder, std::set<EventSetupRecordKey>>),
          recordToPreferred_(new std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription>>),
          recordsWithALooperProxy_(new std::set<EventSetupRecordKey>) {}

    EventSetupProvider::~EventSetupProvider() { forceCacheClear(); }

    std::shared_ptr<EventSetupRecordProvider>& EventSetupProvider::recordProvider(const EventSetupRecordKey& iKey) {
      auto lb = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iKey);
      if (lb == recordKeys_.end() || iKey != *lb) {
        throw cms::Exception("LogicError") << "EventSetupProvider::recordProvider Could not find key\n"
                                           << "Should be impossible. Please contact Framework developer.\n";
      }
      auto index = std::distance(recordKeys_.begin(), lb);
      return recordProviders_[index];
    }

    EventSetupRecordProvider* EventSetupProvider::tryToGetRecordProvider(const EventSetupRecordKey& iKey) {
      auto lb = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iKey);
      if (lb == recordKeys_.end() || iKey != *lb) {
        return nullptr;
      }
      auto index = std::distance(recordKeys_.begin(), lb);
      return recordProviders_[index].get();
    }

    void EventSetupProvider::insert(const EventSetupRecordKey& iKey,
                                    std::unique_ptr<EventSetupRecordProvider> iProvider) {
      auto lb = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iKey);
      auto index = std::distance(recordKeys_.begin(), lb);
      if (lb == recordKeys_.end() || iKey != *lb) {
        recordKeys_.insert(lb, iKey);
        recordProviders_.insert(recordProviders_.begin() + index, std::move(iProvider));
      } else {
        recordProviders_[index] = std::move(iProvider);
      }
    }

    void EventSetupProvider::add(std::shared_ptr<DataProxyProvider> iProvider) {
      assert(iProvider.get() != nullptr);
      dataProviders_->push_back(iProvider);
      if (activityRegistry_) {
        activityRegistry_->postESModuleRegistrationSignal_(iProvider->description());
      }
    }

    void EventSetupProvider::replaceExisting(std::shared_ptr<DataProxyProvider> dataProxyProvider) {
      ParameterSetIDHolder psetID(dataProxyProvider->description().pid_);
      std::set<EventSetupRecordKey> const& keysForPSetID = (*psetIDToRecordKey_)[psetID];
      for (auto const& key : keysForPSetID) {
        recordProvider(key)->resetProxyProvider(psetID, dataProxyProvider);
      }
    }

    void EventSetupProvider::add(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder) {
      assert(iFinder.get() != nullptr);
      finders_->push_back(iFinder);
    }

    using RecordProviders = std::vector<std::shared_ptr<EventSetupRecordProvider>>;
    using RecordToPreferred = std::map<EventSetupRecordKey, EventSetupRecordProvider::DataToPreferredProviderMap>;
    ///find everything made by a DataProxyProvider and add it to the 'preferred' list
    static void preferEverything(const ComponentDescription& iComponent,
                                 const RecordProviders& iRecordProviders,
                                 RecordToPreferred& iReturnValue) {
      //need to get our hands on the actual DataProxyProvider
      bool foundProxyProvider = false;
      for (auto const& recordProvider : iRecordProviders) {
        std::set<ComponentDescription> components = recordProvider->proxyProviderDescriptions();
        if (components.find(iComponent) != components.end()) {
          std::shared_ptr<DataProxyProvider> proxyProv = recordProvider->proxyProvider(*(components.find(iComponent)));
          assert(proxyProv.get());

          std::set<EventSetupRecordKey> records = proxyProv->usingRecords();
          for (auto const& recordKey : records) {
            unsigned int iovIndex = 0;  // Doesn't matter which index is picked, at least 1 should always exist
            DataProxyProvider::KeyedProxies& keyedProxies = proxyProv->keyedProxies(recordKey, iovIndex);
            if (!keyedProxies.unInitialized()) {
              //add them to our output
              EventSetupRecordProvider::DataToPreferredProviderMap& dataToProviderMap = iReturnValue[recordKey];

              for (auto keyedProxy : keyedProxies) {
                EventSetupRecordProvider::DataToPreferredProviderMap::iterator itFind =
                    dataToProviderMap.find(keyedProxy.dataKey_);
                if (itFind != dataToProviderMap.end()) {
                  throw cms::Exception("ESPreferConflict")
                      << "Two providers have been set to be preferred for\n"
                      << keyedProxy.dataKey_.type().name() << " \"" << keyedProxy.dataKey_.name().value() << "\""
                      << "\n the providers are "
                      << "\n 1) type=" << itFind->second.type_ << " label=\"" << itFind->second.label_ << "\""
                      << "\n 2) type=" << iComponent.type_ << " label=\"" << iComponent.label_ << "\""
                      << "\nPlease modify configuration so only one is preferred";
                }
                dataToProviderMap.insert(std::make_pair(keyedProxy.dataKey_, iComponent));
              }
            }
          }
          foundProxyProvider = true;
          break;
        }
      }
      if (!foundProxyProvider) {
        throw cms::Exception("ESPreferNoProvider")
            << "Could not make type=\"" << iComponent.type_ << "\" label=\"" << iComponent.label_
            << "\" a preferred Provider."
            << "\n  Please check spelling of name, or that it was loaded into the job.";
      }
    }

    void EventSetupProvider::determinePreferred() {
      using namespace edm::eventsetup;
      if (preferredProviderInfo_) {
        for (auto const& itInfo : *preferredProviderInfo_) {
          if (itInfo.second.empty()) {
            //want everything
            preferEverything(itInfo.first, recordProviders_, *recordToPreferred_);
          } else {
            for (auto const& itRecData : itInfo.second) {
              std::string recordName = itRecData.first;
              EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
              if (recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag()) {
                throw cms::Exception("ESPreferUnknownRecord")
                    << "Unknown record \"" << recordName
                    << "\" used in es_prefer statement for type=" << itInfo.first.type_ << " label=\""
                    << itInfo.first.label_ << "\"\n Please check spelling.";
                //record not found
              }
              //See if the ProxyProvider provides something for this Record
              EventSetupRecordProvider& recordProviderForKey = *recordProvider(recordKey);

              std::set<ComponentDescription> components = recordProviderForKey.proxyProviderDescriptions();
              std::set<ComponentDescription>::iterator itProxyProv = components.find(itInfo.first);
              if (itProxyProv == components.end()) {
                throw cms::Exception("ESPreferWrongRecord")
                    << "The type=" << itInfo.first.type_ << " label=\"" << itInfo.first.label_
                    << "\" does not provide data for the Record " << recordName;
              }
              //Does it data type exist?
              eventsetup::TypeTag datumType = eventsetup::TypeTag::findType(itRecData.second.first);
              if (datumType == eventsetup::TypeTag()) {
                //not found
                throw cms::Exception("ESPreferWrongDataType")
                    << "The es_prefer statement for type=" << itInfo.first.type_ << " label=\"" << itInfo.first.label_
                    << "\" has the unknown data type \"" << itRecData.second.first << "\""
                    << "\n Please check spelling";
              }
              eventsetup::DataKey datumKey(datumType, itRecData.second.second.c_str());

              //Does the proxyprovider make this?
              std::shared_ptr<DataProxyProvider> proxyProv = recordProviderForKey.proxyProvider(*itProxyProv);
              unsigned int iovIndex = 0;  // Doesn't matter which index is picked, at least 1 should always exist
              const DataProxyProvider::KeyedProxies& keyedProxies = proxyProv->keyedProxies(recordKey, iovIndex);
              if (!keyedProxies.contains(datumKey)) {
                throw cms::Exception("ESPreferWrongData")
                    << "The es_prefer statement for type=" << itInfo.first.type_ << " label=\"" << itInfo.first.label_
                    << "\" specifies the data item \n"
                    << "  type=\"" << itRecData.second.first << "\" label=\"" << itRecData.second.second << "\""
                    << "  which is not provided.  Please check spelling.";
              }

              EventSetupRecordProvider::DataToPreferredProviderMap& dataToProviderMap =
                  (*recordToPreferred_)[recordKey];
              //has another provider already been specified?
              if (dataToProviderMap.end() != dataToProviderMap.find(datumKey)) {
                EventSetupRecordProvider::DataToPreferredProviderMap::iterator itFind =
                    dataToProviderMap.find(datumKey);
                throw cms::Exception("ESPreferConflict")
                    << "Two providers have been set to be preferred for\n"
                    << datumKey.type().name() << " \"" << datumKey.name().value() << "\""
                    << "\n the providers are "
                    << "\n 1) type=" << itFind->second.type_ << " label=\"" << itFind->second.label_ << "\""
                    << "\n 2) type=" << itProxyProv->type_ << " label=\"" << itProxyProv->label_ << "\""
                    << "\nPlease modify configuration so only one is preferred";
              }
              dataToProviderMap.insert(std::make_pair(datumKey, *itProxyProv));
            }
          }
        }
      }
    }

    void EventSetupProvider::finishConfiguration(NumberOfConcurrentIOVs const& numberOfConcurrentIOVs,
                                                 bool& hasNonconcurrentFinder) {
      //we delayed adding finders to the system till here so that everything would be loaded first
      recordToFinders_->clear();
      for (auto& finder : *finders_) {
        if (!finder->concurrentFinder()) {
          hasNonconcurrentFinder = true;
        }

        const std::set<EventSetupRecordKey> recordsUsing = finder->findingForRecords();

        for (auto const& key : recordsUsing) {
          (*recordToFinders_)[key].push_back(finder);

          EventSetupRecordProvider* recProvider = tryToGetRecordProvider(key);
          if (recProvider == nullptr) {
            bool printInfoMsg = true;
            unsigned int nConcurrentIOVs = numberOfConcurrentIOVs.numberOfConcurrentIOVs(key, printInfoMsg);

            //create a provider for this record
            insert(key, std::make_unique<EventSetupRecordProvider>(key, activityRegistry_, nConcurrentIOVs));
            recProvider = tryToGetRecordProvider(key);
          }
          recProvider->addFinder(finder);
        }
      }
      //we've transfered our ownership so this is no longer needed
      finders_.reset();

      //Now handle providers since sources can also be finders and the sources can delay registering
      // their Records and therefore could delay setting up their Proxies
      psetIDToRecordKey_->clear();
      for (auto& dataProxyProvider : *dataProviders_) {
        ParameterSetIDHolder psetID(dataProxyProvider->description().pid_);

        const std::set<EventSetupRecordKey> recordsUsing = dataProxyProvider->usingRecords();
        for (auto const& key : recordsUsing) {
          unsigned int nConcurrentIOVs = numberOfConcurrentIOVs.numberOfConcurrentIOVs(key);
          dataProxyProvider->createKeyedProxies(key, nConcurrentIOVs);

          if (dataProxyProvider->description().isLooper_) {
            recordsWithALooperProxy_->insert(key);
          }

          (*psetIDToRecordKey_)[psetID].insert(key);

          EventSetupRecordProvider* recProvider = tryToGetRecordProvider(key);
          if (recProvider == nullptr) {
            bool printInfoMsg = true;
            nConcurrentIOVs = numberOfConcurrentIOVs.numberOfConcurrentIOVs(key, printInfoMsg);
            //create a provider for this record
            insert(key, std::make_unique<EventSetupRecordProvider>(key, activityRegistry_, nConcurrentIOVs));
            recProvider = tryToGetRecordProvider(key);
          }
          recProvider->add(dataProxyProvider);
        }
      }

      //used for the case where no preferred Providers have been specified for the Record
      static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;

      determinePreferred();

      //For each Provider, find all the Providers it depends on.  If a dependent Provider
      // can not be found pass in an empty list
      //CHANGE: now allow for missing Providers
      for (auto& itRecordProvider : recordProviders_) {
        const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
        RecordToPreferred::const_iterator itRecordFound = recordToPreferred_->find(itRecordProvider->key());
        if (itRecordFound != recordToPreferred_->end()) {
          preferredInfo = &(itRecordFound->second);
        }
        //Give it our list of preferred
        itRecordProvider->usePreferred(*preferredInfo);

        std::set<EventSetupRecordKey> records = itRecordProvider->dependentRecords();
        if (!records.empty()) {
          std::string missingRecords;
          std::vector<std::shared_ptr<EventSetupRecordProvider>> depProviders;
          depProviders.reserve(records.size());
          bool foundAllProviders = true;
          for (auto const& key : records) {
            auto lb = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), key);
            if (lb == recordKeys_.end() || key != *lb) {
              foundAllProviders = false;
              if (missingRecords.empty()) {
                missingRecords = key.name();
              } else {
                missingRecords += ", ";
                missingRecords += key.name();
              }
            } else {
              auto index = std::distance(recordKeys_.begin(), lb);
              depProviders.push_back(recordProviders_[index]);
            }
          }

          if (!foundAllProviders) {
            edm::LogInfo("EventSetupDependency")
                << "The EventSetup record " << itRecordProvider->key().name() << " depends on at least one Record \n ("
                << missingRecords
                << ") which is not present in the job."
                   "\n This may lead to an exception begin thrown during event processing.\n If no exception occurs "
                   "during the job than it is usually safe to ignore this message.";

            //depProviders.clear();
            //NOTE: should provide a warning
          }

          itRecordProvider->setDependentProviders(depProviders);
        }
      }

      auto indices = recordsToProxyIndices();
      for (auto& provider : *dataProviders_) {
        provider->updateLookup(indices);
      }
      dataProviders_.reset();

      mustFinishConfiguration_ = false;
    }

    using Itr = RecordProviders::iterator;
    static void findDependents(const EventSetupRecordKey& iKey,
                               Itr itBegin,
                               Itr itEnd,
                               std::vector<std::shared_ptr<EventSetupRecordProvider>>& oDependents) {
      for (Itr it = itBegin; it != itEnd; ++it) {
        //does it depend on the record in question?
        const std::set<EventSetupRecordKey>& deps = (*it)->dependentRecords();
        if (deps.end() != deps.find(iKey)) {
          oDependents.push_back(*it);
          //now see who is dependent on this record since they will be indirectly dependent on iKey
          findDependents((*it)->key(), itBegin, itEnd, oDependents);
        }
      }
    }

    void EventSetupProvider::resetRecordPlusDependentRecords(const EventSetupRecordKey& iKey) {
      EventSetupRecordProvider* recProvider = tryToGetRecordProvider(iKey);
      if (recProvider == nullptr) {
        return;
      }

      std::vector<std::shared_ptr<EventSetupRecordProvider>> dependents;
      findDependents(iKey, recordProviders_.begin(), recordProviders_.end(), dependents);

      dependents.erase(std::unique(dependents.begin(), dependents.end()), dependents.end());

      recProvider->resetProxies();
      for (auto& d : dependents) {
        d->resetProxies();
      }
    }

    void EventSetupProvider::forceCacheClear() {
      for (auto& recProvider : recordProviders_) {
        if (recProvider) {
          recProvider->resetProxies();
        }
      }
    }

    void EventSetupProvider::checkESProducerSharing(
        ModuleTypeResolverMaker const* resolverMaker,
        EventSetupProvider& precedingESProvider,
        std::set<ParameterSetIDHolder>& sharingCheckDone,
        std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>>& referencedESProducers,
        EventSetupsController& esController) {
      edm::LogVerbatim("EventSetupSharing")
          << "EventSetupProvider::checkESProducerSharing: Checking processes with SubProcess Indexes "
          << subProcessIndex() << " and " << precedingESProvider.subProcessIndex();

      if (referencedESProducers.empty()) {
        for (auto& recProvider : recordProviders_) {
          recProvider->getReferencedESProducers(referencedESProducers);
        }
      }

      // This records whether the configurations of all the DataProxyProviders
      // and finders matches for a particular pair of processes and
      // a particular record and also the records it depends on.
      std::map<EventSetupRecordKey, bool> allComponentsMatch;

      std::map<ParameterSetID, bool> candidateNotRejectedYet;

      // Loop over all the ESProducers which have a DataProxy
      // referenced by any EventSetupRecord in this EventSetupProvider
      for (auto const& iRecord : referencedESProducers) {
        for (auto const& iComponent : iRecord.second) {
          ParameterSetID const& psetID = iComponent->pid_;
          ParameterSetIDHolder psetIDHolder(psetID);
          if (sharingCheckDone.find(psetIDHolder) != sharingCheckDone.end())
            continue;

          bool firstProcessWithThisPSet = false;
          bool precedingHasMatchingPSet = false;

          esController.lookForMatches(psetID,
                                      subProcessIndex_,
                                      precedingESProvider.subProcessIndex_,
                                      firstProcessWithThisPSet,
                                      precedingHasMatchingPSet);

          if (firstProcessWithThisPSet) {
            sharingCheckDone.insert(psetIDHolder);
            allComponentsMatch[iRecord.first] = false;
            continue;
          }

          if (!precedingHasMatchingPSet) {
            allComponentsMatch[iRecord.first] = false;
            continue;
          }

          // An ESProducer that survives to this point is a candidate.
          // It was shared with some other process in the first pass where
          // ESProducers were constructed and one of three possibilities exists:
          //    1) It should not have been shared and a new ESProducer needs
          //    to be created and the proper pointers set.
          //    2) It should have been shared with a different preceding process
          //    in which case some pointers need to be modified.
          //    3) It was originally shared which the correct prior process
          //    in which case nothing needs to be done, but we do need to
          //    do some work to verify that.
          // Make an entry in a map for each of these ESProducers. We
          // will set the value to false if and when we determine
          // the ESProducer cannot be shared between this pair of processes.
          auto iCandidateNotRejectedYet = candidateNotRejectedYet.find(psetID);
          if (iCandidateNotRejectedYet == candidateNotRejectedYet.end()) {
            candidateNotRejectedYet[psetID] = true;
            iCandidateNotRejectedYet = candidateNotRejectedYet.find(psetID);
          }

          // At this point we know that the two processes both
          // have an ESProducer matching the type and label in
          // iComponent and also with exactly the same configuration.
          // And there was not an earlier preceding process
          // where the same instance of the ESProducer could
          // have been shared.  And this ESProducer was referenced
          // by the later process's EventSetupRecord (preferred or
          // or just the only thing that could have made the data).
          // To determine if sharing is allowed, now we need to
          // check if all the DataProxyProviders and all the
          // finders are the same for this record and also for
          // all records this record depends on. And even
          // if this is true, we have to wait until the loop
          // ends because some other DataProxy associated with
          // the ESProducer could write to a different record where
          // the same determination will need to be repeated. Only if
          // all of the the DataProxy's can be shared, can the ESProducer
          // instance be shared across processes.

          if (iCandidateNotRejectedYet->second == true) {
            auto iAllComponentsMatch = allComponentsMatch.find(iRecord.first);
            if (iAllComponentsMatch == allComponentsMatch.end()) {
              // We do not know the value in AllComponents yet and
              // we need it now so we have to do the difficult calculation
              // now.
              bool match = doRecordsMatch(precedingESProvider, iRecord.first, allComponentsMatch, esController);
              allComponentsMatch[iRecord.first] = match;
              iAllComponentsMatch = allComponentsMatch.find(iRecord.first);
            }
            if (!iAllComponentsMatch->second) {
              iCandidateNotRejectedYet->second = false;
            }
          }
        }  // end loop over components used by record
      }    // end loop over records

      // Loop over candidates
      for (auto const& candidate : candidateNotRejectedYet) {
        ParameterSetID const& psetID = candidate.first;
        bool canBeShared = candidate.second;
        if (canBeShared) {
          ParameterSet const& pset = esController.getESProducerPSet(psetID, subProcessIndex_);
          logInfoWhenSharing(pset);
          ParameterSetIDHolder psetIDHolder(psetID);
          sharingCheckDone.insert(psetIDHolder);
          if (esController.isFirstMatch(psetID, subProcessIndex_, precedingESProvider.subProcessIndex_)) {
            continue;  // Proper sharing was already done. Nothing more to do.
          }

          // Need to reset the pointer from the EventSetupRecordProvider to the
          // the DataProxyProvider so these two processes share an ESProducer.

          std::shared_ptr<DataProxyProvider> dataProxyProvider;
          std::set<EventSetupRecordKey> const& keysForPSetID1 = (*precedingESProvider.psetIDToRecordKey_)[psetIDHolder];
          for (auto const& key : keysForPSetID1) {
            dataProxyProvider = precedingESProvider.recordProvider(key)->proxyProvider(psetIDHolder);
            assert(dataProxyProvider);
            break;
          }

          std::set<EventSetupRecordKey> const& keysForPSetID2 = (*psetIDToRecordKey_)[psetIDHolder];
          for (auto const& key : keysForPSetID2) {
            recordProvider(key)->resetProxyProvider(psetIDHolder, dataProxyProvider);
          }
        } else {
          if (esController.isLastMatch(psetID, subProcessIndex_, precedingESProvider.subProcessIndex_)) {
            ParameterSet& pset = esController.getESProducerPSet(psetID, subProcessIndex_);
            ModuleFactory::get()->addTo(esController, *this, pset, resolverMaker, true);
          }
        }
      }
    }

    bool EventSetupProvider::doRecordsMatch(EventSetupProvider& precedingESProvider,
                                            EventSetupRecordKey const& eventSetupRecordKey,
                                            std::map<EventSetupRecordKey, bool>& allComponentsMatch,
                                            EventSetupsController const& esController) {
      // first check if this record matches. If not just return false

      // then find the directly dependent records and iterate over them
      // recursively call this function on them. If they return false
      // set allComponentsMatch to false for them and return false.
      // if they all return true then set allComponents to true
      // and return true.

      if (precedingESProvider.recordsWithALooperProxy_->find(eventSetupRecordKey) !=
          precedingESProvider.recordsWithALooperProxy_->end()) {
        return false;
      }

      if ((*recordToFinders_)[eventSetupRecordKey].size() !=
          (*precedingESProvider.recordToFinders_)[eventSetupRecordKey].size()) {
        return false;
      }

      for (auto const& finder : (*recordToFinders_)[eventSetupRecordKey]) {
        ParameterSetID const& psetID = finder->descriptionForFinder().pid_;
        bool itMatches =
            esController.isMatchingESSource(psetID, subProcessIndex_, precedingESProvider.subProcessIndex_);
        if (!itMatches) {
          return false;
        }
      }

      fillReferencedDataKeys(eventSetupRecordKey);
      precedingESProvider.fillReferencedDataKeys(eventSetupRecordKey);

      std::map<DataKey, ComponentDescription const*> const& dataItems = (*referencedDataKeys_)[eventSetupRecordKey];

      std::map<DataKey, ComponentDescription const*> const& precedingDataItems =
          (*precedingESProvider.referencedDataKeys_)[eventSetupRecordKey];

      if (dataItems.size() != precedingDataItems.size()) {
        return false;
      }

      for (auto const& dataItem : dataItems) {
        auto precedingDataItem = precedingDataItems.find(dataItem.first);
        if (precedingDataItem == precedingDataItems.end()) {
          return false;
        }
        if (dataItem.second->pid_ != precedingDataItem->second->pid_) {
          return false;
        }
        // Check that the configurations match exactly for the ESProducers
        // (We already checked the ESSources above and there should not be
        // any loopers)
        if (!dataItem.second->isSource_ && !dataItem.second->isLooper_) {
          bool itMatches = esController.isMatchingESProducer(
              dataItem.second->pid_, subProcessIndex_, precedingESProvider.subProcessIndex_);
          if (!itMatches) {
            return false;
          }
        }
      }
      EventSetupRecordProvider* recProvider = tryToGetRecordProvider(eventSetupRecordKey);
      if (recProvider != nullptr) {
        std::set<EventSetupRecordKey> dependentRecords = recProvider->dependentRecords();
        for (auto const& dependentRecord : dependentRecords) {
          auto iter = allComponentsMatch.find(dependentRecord);
          if (iter != allComponentsMatch.end()) {
            if (iter->second) {
              continue;
            } else {
              return false;
            }
          }
          bool match = doRecordsMatch(precedingESProvider, dependentRecord, allComponentsMatch, esController);
          allComponentsMatch[dependentRecord] = match;
          if (!match)
            return false;
        }
      }
      return true;
    }

    void EventSetupProvider::fillReferencedDataKeys(EventSetupRecordKey const& eventSetupRecordKey) {
      if (referencedDataKeys_->find(eventSetupRecordKey) != referencedDataKeys_->end())
        return;

      EventSetupRecordProvider* recProvider = tryToGetRecordProvider(eventSetupRecordKey);
      if (recProvider == nullptr) {
        (*referencedDataKeys_)[eventSetupRecordKey];
        return;
      }
      recProvider->fillReferencedDataKeys((*referencedDataKeys_)[eventSetupRecordKey]);
    }

    void EventSetupProvider::resetRecordToProxyPointers() {
      for (auto const& recProvider : recordProviders_) {
        static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;
        const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
        RecordToPreferred::const_iterator itRecordFound = recordToPreferred_->find(recProvider->key());
        if (itRecordFound != recordToPreferred_->end()) {
          preferredInfo = &(itRecordFound->second);
        }
        recProvider->resetRecordToProxyPointers(*preferredInfo);
      }
    }

    void EventSetupProvider::clearInitializationData() {
      preferredProviderInfo_.reset();
      referencedDataKeys_.reset();
      recordToFinders_.reset();
      psetIDToRecordKey_.reset();
      recordToPreferred_.reset();
      recordsWithALooperProxy_.reset();
    }

    void EventSetupProvider::fillRecordsNotAllowingConcurrentIOVs(
        std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
      for (auto const& dataProxyProvider : *dataProviders_) {
        dataProxyProvider->fillRecordsNotAllowingConcurrentIOVs(recordsNotAllowingConcurrentIOVs);
      }
    }

    void EventSetupProvider::setAllValidityIntervals(const IOVSyncValue& iValue) {
      // First loop sets a flag that helps us to not duplicate calls to the
      // same EventSetupRecordProvider setting the IOVs. Dependent records
      // can cause duplicate calls without this protection.
      for (auto& recProvider : recordProviders_) {
        recProvider->initializeForNewSyncValue();
      }

      for (auto& recProvider : recordProviders_) {
        recProvider->setValidityIntervalFor(iValue);
      }
    }

    std::shared_ptr<const EventSetupImpl> EventSetupProvider::eventSetupForInstance(const IOVSyncValue& iValue,
                                                                                    bool& newEventSetupImpl) {
      using IntervalStatus = EventSetupRecordProvider::IntervalStatus;

      // It is important to understand that eventSetupForInstance is a function
      // where only one call is executing at a time (not multiple calls running
      // concurrently). These calls are made in the order determined by the
      // InputSource. One invocation completes and returns before another starts.

      bool needNewEventSetupImpl = false;
      if (eventSetupImpl_.get() == nullptr) {
        needNewEventSetupImpl = true;
      } else {
        for (auto& recProvider : recordProviders_) {
          if (recProvider->intervalStatus() == IntervalStatus::Invalid) {
            if (eventSetupImpl_->validRecord(recProvider->key())) {
              needNewEventSetupImpl = true;
            }
          } else {
            if (recProvider->newIntervalForAnySubProcess()) {
              needNewEventSetupImpl = true;
            }
          }
        }
      }

      if (needNewEventSetupImpl) {
        //cannot use make_shared because constructor is private
        eventSetupImpl_ = std::shared_ptr<EventSetupImpl>(new EventSetupImpl());
        newEventSetupImpl = true;
        eventSetupImpl_->setKeyIters(recordKeys_.begin(), recordKeys_.end());

        for (auto& recProvider : recordProviders_) {
          recProvider->setEventSetupImpl(eventSetupImpl_.get());
        }
      }
      return get_underlying_safe(eventSetupImpl_);
    }

    bool EventSetupProvider::doWeNeedToWaitForIOVsToFinish(IOVSyncValue const& iValue) const {
      for (auto& recProvider : recordProviders_) {
        if (recProvider->doWeNeedToWaitForIOVsToFinish(iValue)) {
          return true;
        }
      }
      return false;
    }

    std::set<ComponentDescription> EventSetupProvider::proxyProviderDescriptions() const {
      typedef std::set<ComponentDescription> Set;
      Set descriptions;

      for (auto const& recProvider : recordProviders_) {
        auto const& d = recProvider->proxyProviderDescriptions();
        descriptions.insert(d.begin(), d.end());
      }
      if (dataProviders_.get()) {
        for (auto const& p : *dataProviders_) {
          descriptions.insert(p->description());
        }
      }

      return descriptions;
    }

    void EventSetupProvider::addRecord(const EventSetupRecordKey& iKey) {
      insert(iKey, std::unique_ptr<EventSetupRecordProvider>());
      eventSetupImpl_->setKeyIters(recordKeys_.begin(), recordKeys_.end());
    }

    void EventSetupProvider::setPreferredProviderInfo(PreferredProviderInfo const& iInfo) {
      preferredProviderInfo_ = std::make_unique<PreferredProviderInfo>(iInfo);
    }

    void EventSetupProvider::fillKeys(std::set<EventSetupRecordKey>& keys) const {
      for (auto const& recProvider : recordProviders_) {
        keys.insert(recProvider->key());
      }
    }

    ESRecordsToProxyIndices EventSetupProvider::recordsToProxyIndices() const {
      ESRecordsToProxyIndices ret(recordKeys_);

      unsigned int index = 0;
      for (const auto& provider : recordProviders_) {
        index = ret.dataKeysInRecord(
            index, provider->key(), provider->registeredDataKeys(), provider->componentsForRegisteredDataKeys());
      }

      return ret;
    }

    //
    // static member functions
    //
    void EventSetupProvider::logInfoWhenSharing(ParameterSet const& iConfiguration) {
      std::string edmtype = iConfiguration.getParameter<std::string>("@module_edm_type");
      std::string modtype = iConfiguration.getParameter<std::string>("@module_type");
      std::string label = iConfiguration.getParameter<std::string>("@module_label");
      edm::LogVerbatim("EventSetupSharing")
          << "Sharing " << edmtype << ": class=" << modtype << " label='" << label << "'";
    }

  }  // namespace eventsetup
}  // namespace edm
