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
#include "boost/bind.hpp"
#include <algorithm>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {
   namespace eventsetup {

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupProvider::EventSetupProvider(unsigned subProcessIndex, const PreferredProviderInfo* iInfo) :
eventSetup_(),
providers_(),
mustFinishConfiguration_(true),
subProcessIndex_(subProcessIndex),
preferredProviderInfo_((0!=iInfo) ? (new PreferredProviderInfo(*iInfo)): 0),
finders_(new std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >() ),
dataProviders_(new std::vector<boost::shared_ptr<DataProxyProvider> >() ),
referencedDataKeys_(new std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription const*> >),
recordToFinders_(new std::map<EventSetupRecordKey, std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> > >),
psetIDToRecordKey_(new std::map<ParameterSetIDHolder, std::set<EventSetupRecordKey> >),
recordToPreferred_(new std::map<EventSetupRecordKey, std::map<DataKey, ComponentDescription> >),
recordsWithALooperProxy_(new std::set<EventSetupRecordKey>)
{
}

// EventSetupProvider::EventSetupProvider(const EventSetupProvider& rhs)
// {
//    // do actual copying here;
// }

EventSetupProvider::~EventSetupProvider()
{
}

//
// assignment operators
//
// const EventSetupProvider& EventSetupProvider::operator=(const EventSetupProvider& rhs)
// {
//   //An exception safe implementation is
//   EventSetupProvider temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EventSetupProvider::insert(const EventSetupRecordKey& iKey, std::auto_ptr<EventSetupRecordProvider> iProvider)
{
   boost::shared_ptr<EventSetupRecordProvider> temp(iProvider.release());
   providers_[iKey] = temp;
   //temp->addRecordTo(*this);
}

void 
EventSetupProvider::add(boost::shared_ptr<DataProxyProvider> iProvider)
{
   assert(iProvider.get() != 0);
   dataProviders_->push_back(iProvider);
}

void 
EventSetupProvider::replaceExisting(boost::shared_ptr<DataProxyProvider> dataProxyProvider)
{
   ParameterSetIDHolder psetID(dataProxyProvider->description().pid_);
   std::set<EventSetupRecordKey> const& keysForPSetID = (*psetIDToRecordKey_)[psetID];
   for (auto const& key : keysForPSetID) {
      boost::shared_ptr<EventSetupRecordProvider> const& recordProvider = providers_[key];
      recordProvider->resetProxyProvider(psetID, dataProxyProvider);
   }
}

void 
EventSetupProvider::add(boost::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   assert(iFinder.get() != 0);
   finders_->push_back(iFinder);
}

typedef std::map<EventSetupRecordKey, boost::shared_ptr<EventSetupRecordProvider> > Providers;
typedef std::map<EventSetupRecordKey, EventSetupRecordProvider::DataToPreferredProviderMap> RecordToPreferred;
///find everything made by a DataProxyProvider and add it to the 'preferred' list
static
void 
preferEverything(const ComponentDescription& iComponent,
                 const Providers& iProviders,
                 RecordToPreferred& iReturnValue)
{
   //need to get our hands on the actual DataProxyProvider
   bool foundProxyProvider = false;
   for(Providers::const_iterator itProvider = iProviders.begin(), itProviderEnd = iProviders.end();
       itProvider!= itProviderEnd;
       ++itProvider) {
      std::set<ComponentDescription> components = itProvider->second->proxyProviderDescriptions();
      if(components.find(iComponent)!= components.end()) {
         boost::shared_ptr<DataProxyProvider> proxyProv = 
         itProvider->second->proxyProvider(*(components.find(iComponent)));
         assert(proxyProv.get());
         
         std::set<EventSetupRecordKey> records = proxyProv->usingRecords();
         for(std::set<EventSetupRecordKey>::iterator itRecord = records.begin(),
             itRecordEnd = records.end();
             itRecord != itRecordEnd;
             ++itRecord){
            const DataProxyProvider::KeyedProxies& keyedProxies = proxyProv->keyedProxies(*itRecord);
            if(!keyedProxies.empty()){
               //add them to our output
               EventSetupRecordProvider::DataToPreferredProviderMap& dataToProviderMap =
               iReturnValue[*itRecord];
               
               for(DataProxyProvider::KeyedProxies::const_iterator itProxy = keyedProxies.begin(),
	           itProxyEnd = keyedProxies.end();
                   itProxy != itProxyEnd;
                   ++itProxy) {
                  EventSetupRecordProvider::DataToPreferredProviderMap::iterator itFind =
                  dataToProviderMap.find(itProxy->first);
                  if(itFind != dataToProviderMap.end()){
                     throw cms::Exception("ESPreferConflict") <<"Two providers have been set to be preferred for\n"
                     <<itProxy->first.type().name()<<" \""<<itProxy->first.name().value()<<"\""
                     <<"\n the providers are "
                     <<"\n 1) type="<<itFind->second.type_<<" label=\""<<itFind->second.label_<<"\""
                     <<"\n 2) type="<<iComponent.type_<<" label=\""<<iComponent.label_<<"\""
                     <<"\nPlease modify configuration so only one is preferred";
                  }
                  dataToProviderMap.insert(std::make_pair(itProxy->first,iComponent));
               }
            }
         }
         foundProxyProvider=true;
         break;
      }
   }
   if(!foundProxyProvider) {
      throw cms::Exception("ESPreferNoProvider")<<"Could not make type=\""<<iComponent.type_
      <<"\" label=\""<<iComponent.label_<<"\" a preferred Provider."<<
      "\n  Please check spelling of name, or that it was loaded into the job.";
   }
}
static
RecordToPreferred determinePreferred(const EventSetupProvider::PreferredProviderInfo* iInfo, 
                                     const Providers& iProviders)
{
   using namespace edm::eventsetup;
   RecordToPreferred returnValue;
   if(0 != iInfo){
      for(EventSetupProvider::PreferredProviderInfo::const_iterator itInfo = iInfo->begin(),
          itInfoEnd = iInfo->end();
          itInfo != itInfoEnd;
          ++itInfo) {
         if(itInfo->second.empty()) {
            //want everything
            preferEverything(itInfo->first, iProviders, returnValue);
         } else {
            for(EventSetupProvider::RecordToDataMap::const_iterator itRecData = itInfo->second.begin(),
	        itRecDataEnd = itInfo->second.end();
                itRecData != itRecDataEnd;
                ++itRecData) {
               std::string recordName= itRecData->first;
               EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
               if(recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag()) {
                  throw cms::Exception("ESPreferUnknownRecord") <<"Unknown record \""<<recordName
                  <<"\" used in es_prefer statement for type="
                  <<itInfo->first.type_<<" label=\""<<itInfo->first.label_
                  <<"\"\n Please check spelling.";
                  //record not found
               }
               //See if the ProxyProvider provides something for this Record
               Providers::const_iterator itRecordProvider = iProviders.find(recordKey);
               assert(itRecordProvider != iProviders.end());
               
               std::set<ComponentDescription> components = itRecordProvider->second->proxyProviderDescriptions();
               std::set<ComponentDescription>::iterator itProxyProv = components.find(itInfo->first);
               if(itProxyProv == components.end()){
                  throw cms::Exception("ESPreferWrongRecord")<<"The type="<<itInfo->first.type_<<" label=\""<<
                  itInfo->first.label_<<"\" does not provide data for the Record "<<recordName;
               }
               //Does it data type exist?
               eventsetup::TypeTag datumType = eventsetup::TypeTag::findType(itRecData->second.first);
               if(datumType == eventsetup::TypeTag()) {
                  //not found
                  throw cms::Exception("ESPreferWrongDataType")<<"The es_prefer statement for type="<<itInfo->first.type_<<" label=\""<<
                  itInfo->first.label_<<"\" has the unknown data type \""<<itRecData->second.first<<"\""
                  <<"\n Please check spelling";
               }
               eventsetup::DataKey datumKey(datumType, itRecData->second.second.c_str());
               
               //Does the proxyprovider make this?
               boost::shared_ptr<DataProxyProvider> proxyProv = 
                  itRecordProvider->second->proxyProvider(*itProxyProv);
               const DataProxyProvider::KeyedProxies& keyedProxies = proxyProv->keyedProxies(recordKey);
               if(std::find_if(keyedProxies.begin(), keyedProxies.end(), 
                                boost::bind(std::equal_to<DataKey>(), datumKey, boost::bind(&DataProxyProvider::KeyedProxies::value_type::first,_1))) ==
                   keyedProxies.end()){
                  throw cms::Exception("ESPreferWrongData")<<"The es_prefer statement for type="<<itInfo->first.type_<<" label=\""<<
                  itInfo->first.label_<<"\" specifies the data item \n"
                  <<"  type=\""<<itRecData->second.first<<"\" label=\""<<itRecData->second.second<<"\""
                  <<"  which is not provided.  Please check spelling.";
               }
               
               EventSetupRecordProvider::DataToPreferredProviderMap& dataToProviderMap
                  =returnValue[recordKey];
               //has another provider already been specified?
               if(dataToProviderMap.end() != dataToProviderMap.find(datumKey)) {
                  EventSetupRecordProvider::DataToPreferredProviderMap::iterator itFind =
                  dataToProviderMap.find(datumKey);
                  throw cms::Exception("ESPreferConflict") <<"Two providers have been set to be preferred for\n"
                  <<datumKey.type().name()<<" \""<<datumKey.name().value()<<"\""
                  <<"\n the providers are "
                  <<"\n 1) type="<<itFind->second.type_<<" label=\""<<itFind->second.label_<<"\""
                  <<"\n 2) type="<<itProxyProv->type_<<" label=\""<<itProxyProv->label_<<"\""
                  <<"\nPlease modify configuration so only one is preferred";
               }
               dataToProviderMap.insert(std::make_pair(datumKey,*itProxyProv));
            }
         }
      }
   }
   return returnValue;
}

void
EventSetupProvider::finishConfiguration()
{   
   //we delayed adding finders to the system till here so that everything would be loaded first
   recordToFinders_->clear();
   for(std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >::iterator itFinder=finders_->begin(),
       itEnd = finders_->end();
       itFinder != itEnd;
       ++itFinder) {
      typedef std::set<EventSetupRecordKey> Keys;
      const Keys recordsUsing = (*itFinder)->findingForRecords();
      
      for(Keys::const_iterator itKey = recordsUsing.begin(), itKeyEnd = recordsUsing.end();
          itKey != itKeyEnd;
          ++itKey) {
        (*recordToFinders_)[*itKey].push_back(*itFinder);
         Providers::iterator itFound = providers_.find(*itKey);
         if(providers_.end() == itFound) {
            //create a provider for this record
            insert(*itKey, EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(*itKey));
            itFound = providers_.find(*itKey);
         }
         itFound->second->addFinder(*itFinder);
      }      
   }
   //we've transfered our ownership so this is no longer needed
   finders_.reset();

   //Now handle providers since sources can also be finders and the sources can delay registering
   // their Records and therefore could delay setting up their Proxies
   psetIDToRecordKey_->clear();
   typedef std::set<EventSetupRecordKey> Keys;
   for(std::vector<boost::shared_ptr<DataProxyProvider> >::iterator itProvider=dataProviders_->begin(),
       itEnd = dataProviders_->end();
       itProvider != itEnd;
       ++itProvider) {
      
      ParameterSetIDHolder psetID((*itProvider)->description().pid_);

      const Keys recordsUsing = (*itProvider)->usingRecords();
      
      for(Keys::const_iterator itKey = recordsUsing.begin(), itKeyEnd = recordsUsing.end();
          itKey != itKeyEnd;
          ++itKey) {

         if ((*itProvider)->description().isLooper_) {
            recordsWithALooperProxy_->insert(*itKey);
         }

         (*psetIDToRecordKey_)[psetID].insert(*itKey);

         Providers::iterator itFound = providers_.find(*itKey);
         if(providers_.end() == itFound) {
            //create a provider for this record
            insert(*itKey, EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(*itKey));
            itFound = providers_.find(*itKey);
         }
         itFound->second->add(*itProvider);
      }
   }
   dataProviders_.reset();
   
   //used for the case where no preferred Providers have been specified for the Record
   static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;
   
   *recordToPreferred_ = determinePreferred(preferredProviderInfo_.get(),providers_);
   //For each Provider, find all the Providers it depends on.  If a dependent Provider
   // can not be found pass in an empty list
   //CHANGE: now allow for missing Providers
   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
        itProvider != itProviderEnd;
        ++itProvider) {
      const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
      RecordToPreferred::const_iterator itRecordFound = recordToPreferred_->find(itProvider->first);
      if(itRecordFound != recordToPreferred_->end()) {
         preferredInfo = &(itRecordFound->second);
      }
      //Give it our list of preferred 
      itProvider->second->usePreferred(*preferredInfo);
      
      std::set<EventSetupRecordKey> records = itProvider->second->dependentRecords();
      if(records.size() != 0) {
         std::string missingRecords;
         std::vector<boost::shared_ptr<EventSetupRecordProvider> > depProviders;
         depProviders.reserve(records.size());
         bool foundAllProviders = true;
         for(std::set<EventSetupRecordKey>::iterator itRecord = records.begin(),
              itRecordEnd = records.end();
              itRecord != itRecordEnd;
              ++itRecord) {
            Providers::iterator itFound = providers_.find(*itRecord);
            if(itFound == providers_.end()) {
               foundAllProviders = false;
               if(missingRecords.size() == 0) {
                 missingRecords = itRecord->name();
               } else {
                 missingRecords += ", ";
                 missingRecords += itRecord->name();
               }
               //break;
            } else {
               depProviders.push_back(itFound->second);
            }
         }
         
         if(!foundAllProviders) {
            edm::LogInfo("EventSetupDependency")<<"The EventSetup record "<<itProvider->second->key().name()
            <<" depends on at least one Record \n ("<<missingRecords<<") which is not present in the job."
           "\n This may lead to an exception begin thrown during event processing.\n If no exception occurs during the job than it is usually safe to ignore this message.";

            //depProviders.clear();
           //NOTE: should provide a warning
         }
          
         itProvider->second->setDependentProviders(depProviders);
      }
   }
   mustFinishConfiguration_ = false;
}

typedef std::map<EventSetupRecordKey, boost::shared_ptr<EventSetupRecordProvider> > Providers;
typedef Providers::iterator Itr;
static
void
findDependents(const EventSetupRecordKey& iKey,
               Itr itBegin,
               Itr itEnd,
               std::vector<boost::shared_ptr<EventSetupRecordProvider> >& oDependents)
{
  
  for(Itr it = itBegin; it != itEnd; ++it) {
    //does it depend on the record in question?
    const std::set<EventSetupRecordKey>& deps = it->second->dependentRecords();
    if(deps.end() != deps.find(iKey)) {
      oDependents.push_back(it->second);
      //now see who is dependent on this record since they will be indirectly dependent on iKey
      findDependents(it->first, itBegin, itEnd, oDependents);
    }
  }
}
               
void 
EventSetupProvider::resetRecordPlusDependentRecords(const EventSetupRecordKey& iKey)
{
  Providers::iterator itFind = providers_.find(iKey);
  if(itFind == providers_.end()) {
    return;
  }

  
  std::vector<boost::shared_ptr<EventSetupRecordProvider> > dependents;
  findDependents(iKey, providers_.begin(), providers_.end(), dependents);

  dependents.erase(std::unique(dependents.begin(),dependents.end()), dependents.end());
  
  itFind->second->resetProxies();
  for_all(dependents,
                boost::bind(&EventSetupRecordProvider::resetProxies,
                            _1));
}

void 
EventSetupProvider::forceCacheClear()
{
   for(Providers::iterator it=providers_.begin(), itEnd = providers_.end();
       it != itEnd;
       ++it) {
      it->second->resetProxies();
   }
}

void
EventSetupProvider::checkESProducerSharing(EventSetupProvider& precedingESProvider,
                                           std::set<ParameterSetIDHolder>& sharingCheckDone,
                                           std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> >& referencedESProducers,
                                           EventSetupsController& esController) {

   edm::LogVerbatim("EventSetupSharing") << "EventSetupProvider::checkESProducerSharing: Checking processes with SubProcess Indexes "
                                         << subProcessIndex() << " and " << precedingESProvider.subProcessIndex();

   if (referencedESProducers.empty()) {
      for (auto const& recordProvider : providers_) {
         recordProvider.second->getReferencedESProducers(referencedESProducers);
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
         if (sharingCheckDone.find(psetIDHolder) != sharingCheckDone.end()) continue;

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
         // by the later process's EventSetupRecord (prefered or
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
               bool match = doRecordsMatch(precedingESProvider,
                                           iRecord.first,
                                           allComponentsMatch,
                                           esController);
               allComponentsMatch[iRecord.first] = match;
               iAllComponentsMatch = allComponentsMatch.find(iRecord.first);            
            }
            if (!iAllComponentsMatch->second) {
              iCandidateNotRejectedYet->second = false;
            }
         }
      }  // end loop over components used by record
   } // end loop over records

   // Loop over candidates
   for (auto const& candidate : candidateNotRejectedYet) {
      ParameterSetID const& psetID = candidate.first;
      bool canBeShared = candidate.second;
      if (canBeShared) {
         ParameterSet const& pset = *esController.getESProducerPSet(psetID, subProcessIndex_);
         logInfoWhenSharing(pset);
         ParameterSetIDHolder psetIDHolder(psetID);
         sharingCheckDone.insert(psetIDHolder);
         if (esController.isFirstMatch(psetID,
                                       subProcessIndex_,
                                       precedingESProvider.subProcessIndex_)) {
            continue;  // Proper sharing was already done. Nothing more to do.
         }

         // Need to reset the pointer from the EventSetupRecordProvider to the
         // the DataProxyProvider so these two processes share an ESProducer.

         boost::shared_ptr<DataProxyProvider> dataProxyProvider;
         std::set<EventSetupRecordKey> const& keysForPSetID1 = (*precedingESProvider.psetIDToRecordKey_)[psetIDHolder];
         for (auto const& key : keysForPSetID1) {
            boost::shared_ptr<EventSetupRecordProvider> const& recordProvider = precedingESProvider.providers_[key];
            dataProxyProvider = recordProvider->proxyProvider(psetIDHolder);
            assert(dataProxyProvider);
            break;
         }

         std::set<EventSetupRecordKey> const& keysForPSetID2 = (*psetIDToRecordKey_)[psetIDHolder];
         for (auto const& key : keysForPSetID2) {
            boost::shared_ptr<EventSetupRecordProvider> const& recordProvider = providers_[key];
            recordProvider->resetProxyProvider(psetIDHolder, dataProxyProvider);
         }
      } else {
         if (esController.isLastMatch(psetID,
                                      subProcessIndex_,
                                      precedingESProvider.subProcessIndex_)) {
            

            ParameterSet const& pset = *esController.getESProducerPSet(psetID, subProcessIndex_);
            ModuleFactory::get()->addTo(esController,
                                        *this,
                                        pset,
                                        true);

         }
      }
   }
}

bool
EventSetupProvider::doRecordsMatch(EventSetupProvider & precedingESProvider,
                                   EventSetupRecordKey const& eventSetupRecordKey,
                                   std::map<EventSetupRecordKey, bool> & allComponentsMatch,
                                   EventSetupsController const& esController) {
   // first check if this record matches. If not just return false

   // then find the directly dependent records and iterate over them
   // recursively call this function on them. If they return false
   // set allComponentsMatch to false for them and return false.
   // if they all return true then set allComponents to true
   // and return true.

   if (precedingESProvider.recordsWithALooperProxy_->find(eventSetupRecordKey) != precedingESProvider.recordsWithALooperProxy_->end()) {
      return false;
   }

   if ((*recordToFinders_)[eventSetupRecordKey].size() != (*precedingESProvider.recordToFinders_)[eventSetupRecordKey].size()) {
      return false;
   }

   for (auto const& finder : (*recordToFinders_)[eventSetupRecordKey]) {
      ParameterSetID const& psetID = finder->descriptionForFinder().pid_;
      bool itMatches = esController.isMatchingESSource(psetID,
                                                       subProcessIndex_,
                                                       precedingESProvider.subProcessIndex_);
      if (!itMatches) {
         return false;
      }
   }

   fillReferencedDataKeys(eventSetupRecordKey);
   precedingESProvider.fillReferencedDataKeys(eventSetupRecordKey);

   std::map<DataKey, ComponentDescription const*> const& dataItems = (*referencedDataKeys_)[eventSetupRecordKey];

   std::map<DataKey, ComponentDescription const*> const& precedingDataItems = (*precedingESProvider.referencedDataKeys_)[eventSetupRecordKey];

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
         bool itMatches = esController.isMatchingESProducer(dataItem.second->pid_,
                                                            subProcessIndex_,
                                                            precedingESProvider.subProcessIndex_);
         if (!itMatches) {
            return false;
         }
      }
   }
   Providers::iterator itFound = providers_.find(eventSetupRecordKey);
   if (itFound != providers_.end()) {
      std::set<EventSetupRecordKey> dependentRecords = itFound->second->dependentRecords();
      for (auto const& dependentRecord : dependentRecords) {
         auto iter = allComponentsMatch.find(dependentRecord);
         if (iter != allComponentsMatch.end()) {
            if (iter->second) {
               continue;
            } else {
               return false;
            }
         }
         bool match = doRecordsMatch(precedingESProvider,
                                     dependentRecord,
                                     allComponentsMatch,
                                     esController);            
         allComponentsMatch[dependentRecord] = match;
         if (!match) return false;
      }
   }
   return true;
}

void
EventSetupProvider::fillReferencedDataKeys(EventSetupRecordKey const& eventSetupRecordKey) {

   if (referencedDataKeys_->find(eventSetupRecordKey) != referencedDataKeys_->end()) return;

   auto recordProvider = providers_.find(eventSetupRecordKey);
   if (recordProvider == providers_.end()) {
      (*referencedDataKeys_)[eventSetupRecordKey];
      return;
   }
   if (recordProvider->second) {
      recordProvider->second->fillReferencedDataKeys((*referencedDataKeys_)[eventSetupRecordKey]);
   }
}

void
EventSetupProvider::resetRecordToProxyPointers() {
   for (auto const& recordProvider : providers_) {
      static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;
      const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
      RecordToPreferred::const_iterator itRecordFound = recordToPreferred_->find(recordProvider.first);
      if(itRecordFound != recordToPreferred_->end()) {
         preferredInfo = &(itRecordFound->second);
      }
      recordProvider.second->resetRecordToProxyPointers(*preferredInfo);
   }
}

void
EventSetupProvider::clearInitializationData() {
   preferredProviderInfo_.reset();
   referencedDataKeys_.reset();
   recordToFinders_.reset();
   psetIDToRecordKey_.reset();
   recordToPreferred_.reset();
   recordsWithALooperProxy_.reset();
}

void
EventSetupProvider::addRecordToEventSetup(EventSetupRecord& iRecord) {
   iRecord.setEventSetup(&eventSetup_);
   eventSetup_.add(iRecord);
}
      
//
// const member functions
//
EventSetup const&
EventSetupProvider::eventSetupForInstance(const IOVSyncValue& iValue)
{
   eventSetup_.setIOVSyncValue(iValue);

   eventSetup_.clear();

   // In a cmsRun job this does nothing because the EventSetupsController
   // will have already called finishConfiguration, but some tests will
   // call finishConfiguration here.
   if(mustFinishConfiguration_) {
      finishConfiguration();
   }

   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
        itProvider != itProviderEnd;
        ++itProvider) {
      itProvider->second->addRecordToIfValid(*this, iValue);
   }   
   return eventSetup_;
}

namespace {
   struct InsertAll : public std::unary_function< const std::set<ComponentDescription>&, void>{
      
      typedef std::set<ComponentDescription> Set;
      Set* container_;
      InsertAll(Set& iSet) : container_(&iSet) {}
      void operator()(const Set& iSet) {
         container_->insert(iSet.begin(), iSet.end());
      }
   };
}
std::set<ComponentDescription> 
EventSetupProvider::proxyProviderDescriptions() const
{
   using boost::bind;
   typedef std::set<ComponentDescription> Set;
   Set descriptions;

   for_all(providers_,
                 bind(InsertAll(descriptions),
                      bind(&EventSetupRecordProvider::proxyProviderDescriptions,
                           bind(&Providers::value_type::second,_1))));
   if(dataProviders_.get()) {
      for(std::vector<boost::shared_ptr<DataProxyProvider> >::const_iterator it = dataProviders_->begin(),
          itEnd = dataProviders_->end();
          it != itEnd;
          ++it) {
         descriptions.insert((*it)->description());
      }
         
   }
                       
   return descriptions;
}

//
// static member functions
//
void EventSetupProvider::logInfoWhenSharing(ParameterSet const& iConfiguration) {

   std::string edmtype = iConfiguration.getParameter<std::string>("@module_edm_type");
   std::string modtype = iConfiguration.getParameter<std::string>("@module_type");
   std::string label = iConfiguration.getParameter<std::string>("@module_label");
   edm::LogVerbatim("EventSetupSharing") << "Sharing " << edmtype << ": class=" << modtype << " label='" << label << "'";
}
   }
}
