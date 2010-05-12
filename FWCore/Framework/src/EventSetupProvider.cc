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

// user include files
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
EventSetupProvider::EventSetupProvider(const PreferredProviderInfo* iInfo) :
eventSetup_(),
providers_(),
mustFinishConfiguration_(true),
preferredProviderInfo_((0!=iInfo) ? (new PreferredProviderInfo(*iInfo)): 0),
finders_(new std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >() )
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
   mustFinishConfiguration_ = true;
   assert(&(*iProvider) != 0);
   typedef std::set<EventSetupRecordKey> Keys;
   const Keys recordsUsing = iProvider->usingRecords();

   for(Keys::const_iterator itKey = recordsUsing.begin(), itKeyEnd = recordsUsing.end();
       itKey != itKeyEnd;
       ++itKey) {
      Providers::iterator itFound = providers_.find(*itKey);
      if(providers_.end() == itFound) {
         //create a provider for this record
         insert(*itKey, EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(*itKey));
         itFound = providers_.find(*itKey);
      }
      itFound->second->add(iProvider);
   }
}


void 
EventSetupProvider::add(boost::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   mustFinishConfiguration_ = true;
   assert(&(*iFinder) != 0);
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
   for(std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >::iterator itFinder=finders_->begin(),
       itEnd = finders_->end();
       itFinder != itEnd;
       ++itFinder) {
      typedef std::set<EventSetupRecordKey> Keys;
      const Keys recordsUsing = (*itFinder)->findingForRecords();
      
      for(Keys::const_iterator itKey = recordsUsing.begin(), itKeyEnd = recordsUsing.end();
          itKey != itKeyEnd;
          ++itKey) {
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
   
   
   //used for the case where no preferred Providers have been specified for the Record
   static const EventSetupRecordProvider::DataToPreferredProviderMap kEmptyMap;
   
   RecordToPreferred recordToPreferred = determinePreferred(preferredProviderInfo_.get(),providers_);
   //For each Provider, find all the Providers it depends on.  If a dependent Provider
   // can not be found pass in an empty list
   //CHANGE: now allow for missing Providers
   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
        itProvider != itProviderEnd;
        ++itProvider) {
      const EventSetupRecordProvider::DataToPreferredProviderMap* preferredInfo = &kEmptyMap;
      RecordToPreferred::const_iterator itRecordFound = recordToPreferred.find(itProvider->first);
      if(itRecordFound != recordToPreferred.end()) {
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

//
// const member functions
//
EventSetup const&
EventSetupProvider::eventSetupForInstance(const IOVSyncValue& iValue)
{
   eventSetup_.setIOVSyncValue(iValue);

   eventSetup_.clear();
   if(mustFinishConfiguration_) {
      const_cast<EventSetupProvider*>(this)->finishConfiguration();
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
                       
   return descriptions;
}

//
// static member functions
//
   }
}
