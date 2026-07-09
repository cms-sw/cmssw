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

#include "FWCore/Framework/interface/EventSetupProvider.h"

// system include files
#include <algorithm>
#include <cassert>
#include <iterator>
#include <unordered_set>

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/NumberOfConcurrentIOVs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace eventsetup {

    EventSetupProvider::EventSetupProvider(ActivityRegistry const* activityRegistry, const PreferredProviderInfo* iInfo)
        : activityRegistry_(activityRegistry),
          preferredProviderInfo_((nullptr != iInfo) ? (new PreferredProviderInfo(*iInfo)) : nullptr),
          finders_(new std::vector<std::shared_ptr<EventSetupRecordIntervalFinder>>()),
          dataProviders_(new std::vector<std::shared_ptr<ESProductResolverProvider>>()),
          recordToPreferred_(new std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription>>) {}

    EventSetupProvider::~EventSetupProvider() {}

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

    EventSetupRecordProvider const* EventSetupProvider::tryToGetRecordProvider(const EventSetupRecordKey& iKey) const {
      auto lb = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iKey);
      if (lb == recordKeys_.end() || iKey != *lb) {
        return nullptr;
      }
      auto index = std::distance(recordKeys_.begin(), lb);
      return recordProviders_[index].get();
    }

    void EventSetupProvider::fillAllESProductResolverProviders(
        std::vector<ESProductResolverProvider const*>& allESProductResolverProviders) const {
      std::unordered_set<unsigned int> componentIDs;
      for (auto const& recordProvider : recordProviders_) {
        recordProvider->fillAllESProductResolverProviders(allESProductResolverProviders, componentIDs);
      }
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

    void EventSetupProvider::add(std::shared_ptr<ESProductResolverProvider> iProvider) {
      assert(iProvider.get() != nullptr);
      dataProviders_->push_back(iProvider);
      if (activityRegistry_) {
        activityRegistry_->postESModuleRegistrationSignal_.emit(iProvider->description());
      }
    }

    void EventSetupProvider::add(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder) {
      assert(iFinder.get() != nullptr);
      finders_->push_back(iFinder);
    }

    using RecordProviders = std::vector<std::shared_ptr<EventSetupRecordProvider>>;
    using RecordToPreferred = std::map<EventSetupRecordKey, EventSetupRecordProvider::DataToPreferredProviderMap>;
    ///find everything made by a ESProductResolverProvider and add it to the 'preferred' list
    static void preferEverything(const ComponentDescription& iComponent,
                                 const RecordProviders& iRecordProviders,
                                 RecordToPreferred& iReturnValue) {
      //need to get our hands on the actual ESProductResolverProvider
      bool foundResolverProvider = false;
      for (auto const& recordProvider : iRecordProviders) {
        std::set<ComponentDescription> components = recordProvider->resolverProviderDescriptions();
        if (components.find(iComponent) != components.end()) {
          std::shared_ptr<ESProductResolverProvider> resolverProv =
              recordProvider->resolverProvider(*(components.find(iComponent)));
          assert(resolverProv.get());

          std::set<EventSetupRecordKey> records = resolverProv->usingRecords();
          for (auto const& recordKey : records) {
            unsigned int iovIndex = 0;  // Doesn't matter which index is picked, at least 1 should always exist
            ESProductResolverProvider::KeyedResolvers& keyedResolvers =
                resolverProv->keyedResolvers(recordKey, iovIndex);
            if (!keyedResolvers.unInitialized()) {
              //add them to our output
              EventSetupRecordProvider::DataToPreferredProviderMap& dataToProviderMap = iReturnValue[recordKey];

              for (auto keyedResolver : keyedResolvers) {
                EventSetupRecordProvider::DataToPreferredProviderMap::iterator itFind =
                    dataToProviderMap.find(keyedResolver.dataKey_);
                if (itFind != dataToProviderMap.end()) {
                  throw cms::Exception("ESPreferConflict")
                      << "Two providers have been set to be preferred for\n"
                      << keyedResolver.dataKey_.type().name() << " \"" << keyedResolver.dataKey_.name().value() << "\""
                      << "\n the providers are "
                      << "\n 1) type=" << itFind->second.type_ << " label=\"" << itFind->second.label_ << "\""
                      << "\n 2) type=" << iComponent.type_ << " label=\"" << iComponent.label_ << "\""
                      << "\nPlease modify configuration so only one is preferred";
                }
                dataToProviderMap.insert(std::make_pair(keyedResolver.dataKey_, iComponent));
              }
            }
          }
          foundResolverProvider = true;
          break;
        }
      }
      if (!foundResolverProvider) {
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
              //See if the ResolverProvider provides something for this Record
              EventSetupRecordProvider& recordProviderForKey = *recordProvider(recordKey);

              std::set<ComponentDescription> components = recordProviderForKey.resolverProviderDescriptions();
              std::set<ComponentDescription>::iterator itResolverProv = components.find(itInfo.first);
              if (itResolverProv == components.end()) {
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

              //Does the resolverprovider make this?
              std::shared_ptr<ESProductResolverProvider> resolverProv =
                  recordProviderForKey.resolverProvider(*itResolverProv);
              unsigned int iovIndex = 0;  // Doesn't matter which index is picked, at least 1 should always exist
              const ESProductResolverProvider::KeyedResolvers& keyedResolvers =
                  resolverProv->keyedResolvers(recordKey, iovIndex);
              if (!keyedResolvers.contains(datumKey)) {
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
                    << "\n 2) type=" << itResolverProv->type_ << " label=\"" << itResolverProv->label_ << "\""
                    << "\nPlease modify configuration so only one is preferred";
              }
              dataToProviderMap.insert(std::make_pair(datumKey, *itResolverProv));
            }
          }
        }
      }
    }

    void EventSetupProvider::finishConfiguration(NumberOfConcurrentIOVs const& numberOfConcurrentIOVs) {
      //we delayed adding finders to the system till here so that everything would be loaded first
      for (auto& finder : *finders_) {
        const std::set<EventSetupRecordKey> recordsUsing = finder->findingForRecords();

        for (auto const& key : recordsUsing) {
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
      // their Records and therefore could delay setting up their Resolvers
      for (auto& productResolverProvider : *dataProviders_) {
        const std::set<EventSetupRecordKey> recordsUsing = productResolverProvider->usingRecords();
        for (auto const& key : recordsUsing) {
          unsigned int nConcurrentIOVs = numberOfConcurrentIOVs.numberOfConcurrentIOVs(key);
          productResolverProvider->createKeyedResolvers(key, nConcurrentIOVs);

          EventSetupRecordProvider* recProvider = tryToGetRecordProvider(key);
          if (recProvider == nullptr) {
            bool printInfoMsg = true;
            nConcurrentIOVs = numberOfConcurrentIOVs.numberOfConcurrentIOVs(key, printInfoMsg);
            //create a provider for this record
            insert(key, std::make_unique<EventSetupRecordProvider>(key, activityRegistry_, nConcurrentIOVs));
            recProvider = tryToGetRecordProvider(key);
          }
          recProvider->add(productResolverProvider);
        }
      }

      //used for the case where no preferred Providers have been specified for the Record
      static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;

      determinePreferred();

      std::map<EventSetupRecordKey, std::shared_ptr<EventSetupRecordIntervalFinder>> keyToFinders;
      for (auto& itRecordProvider : recordProviders_) {
        const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
        RecordToPreferred::const_iterator itRecordFound = recordToPreferred_->find(itRecordProvider->key());
        if (itRecordFound != recordToPreferred_->end()) {
          preferredInfo = &(itRecordFound->second);
        }
        //Give it our list of preferred
        itRecordProvider->usePreferred(*preferredInfo);
        keyToFinders.insert(std::make_pair(itRecordProvider->key(), itRecordProvider->finalizeFinder()));
      }
      //For each Provider, find all the Providers it depends on.
      for (auto& itRecordProvider : recordProviders_) {
        itRecordProvider->setSupportingFinders(keyToFinders);
      }

      dataProviders_.reset();
    }

    using Itr = RecordProviders::iterator;
    static void findDependents(const EventSetupRecordKey& iKey,
                               Itr itBegin,
                               Itr itEnd,
                               std::vector<std::shared_ptr<EventSetupRecordProvider>>& oDependents) {
      for (Itr it = itBegin; it != itEnd; ++it) {
        //does it depend on the record in question?
        const std::set<EventSetupRecordKey>& deps = (*it)->supportingRecords();
        if (deps.end() != deps.find(iKey)) {
          oDependents.push_back(*it);
          //now see who is dependent on this record since they will be indirectly dependent on iKey
          findDependents((*it)->key(), itBegin, itEnd, oDependents);
        }
      }
    }

    std::vector<EventSetupRecordKey> EventSetupProvider::resetRecordPlusDependentRecords(
        const EventSetupRecordKey& iKey) {
      EventSetupRecordProvider* recProvider = tryToGetRecordProvider(iKey);
      if (recProvider == nullptr) {
        return {};
      }

      std::vector<std::shared_ptr<EventSetupRecordProvider>> dependents;
      findDependents(iKey, recordProviders_.begin(), recordProviders_.end(), dependents);

      dependents.erase(std::unique(dependents.begin(), dependents.end()), dependents.end());

      std::vector<EventSetupRecordKey> ret;
      ret.reserve(dependents.size());
      recProvider->resetResolvers();
      for (auto& d : dependents) {
        d->resetResolvers();
        ret.push_back(d->key());
      }
      return ret;
    }

    void EventSetupProvider::updateLookup() {
      auto indices = recordsToResolverIndices();
      for (auto& recordProvider : recordProviders_) {
        recordProvider->updateLookup(indices);
      }
    }

    void EventSetupProvider::clearInitializationData() {
      preferredProviderInfo_.reset();
      recordToPreferred_.reset();
    }

    void EventSetupProvider::fillRecordsNotAllowingConcurrentIOVs(
        std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
      for (auto const& productResolverProvider : *dataProviders_) {
        productResolverProvider->fillRecordsNotAllowingConcurrentIOVs(recordsNotAllowingConcurrentIOVs);
      }
    }

    std::shared_ptr<EventSetupImpl> EventSetupProvider::cachedEventSetup(bool newEventSetupImpl) {
      if (newEventSetupImpl or !eventSetupImpl_) {
        //cannot use make_shared because constructor is private
        eventSetupImpl_ = std::shared_ptr<EventSetupImpl>(new EventSetupImpl());
        eventSetupImpl_->setKeyIters(recordKeys_.begin(), recordKeys_.end());
      }
      return get_underlying_safe(eventSetupImpl_);
    }

    std::set<ComponentDescription> EventSetupProvider::resolverProviderDescriptions() const {
      typedef std::set<ComponentDescription> Set;
      Set descriptions;

      for (auto const& recProvider : recordProviders_) {
        auto const& d = recProvider->resolverProviderDescriptions();
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

    ESRecordsToProductResolverIndices EventSetupProvider::recordsToResolverIndices() const {
      ESRecordsToProductResolverIndices ret(recordKeys_);

      unsigned int index = 0;
      for (const auto& provider : recordProviders_) {
        index = ret.dataKeysInRecord(index,
                                     provider->key(),
                                     provider->registeredDataKeys(),
                                     provider->componentsForRegisteredDataKeys(),
                                     provider->produceMethodIDsForRegisteredDataKeys());
      }

      return ret;
    }

  }  // namespace eventsetup
}  // namespace edm
