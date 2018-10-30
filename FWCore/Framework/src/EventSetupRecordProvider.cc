// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProvider
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Fri Mar 25 19:12:03 EST 2005
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "make_shared_noexcept_false.h"




//
// constants, enums and typedefs
//

namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey) :
    record_(iKey),
    key_(iKey),
    validityInterval_(), finder_(), providers_(),
    multipleFinders_(new std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>()),
    lastSyncWasBeginOfRun_(true)
{
}

// EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordProvider& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const EventSetupRecordProvider& EventSetupRecordProvider::operator=(const EventSetupRecordProvider& rhs)
// {
//   //An exception safe implementation is
//   EventSetupRecordProvider temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EventSetupRecordProvider::add(std::shared_ptr<DataProxyProvider> iProvider)
{
   assert(iProvider->isUsingRecord(key_));
   edm::propagate_const<std::shared_ptr<DataProxyProvider>> pProvider(iProvider);
   assert(!search_all(providers_, pProvider));
   providers_.emplace_back(iProvider);
}

void 
EventSetupRecordProvider::addFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   auto oldFinder = finder();
   finder_ = iFinder;
   if (nullptr != multipleFinders_.get()) {
     multipleFinders_->emplace_back(iFinder);
   } else {
     //dependent records set there finders after the multipleFinders_ has been released
     // but they also have never had a finder set
     if(nullptr != oldFinder.get()) {
       cms::Exception("EventSetupMultipleSources")<<"An additional source has been added to the Record "
       <<key_.name()<<"'\n"
       <<"after all the other sources have been dealt with.  This is a logic error, please send email to the framework group.";
     }
   }
}
void
EventSetupRecordProvider::setValidityInterval(const ValidityInterval& iInterval)
{
   validityInterval_ = iInterval;
}

void 
EventSetupRecordProvider::setDependentProviders(const std::vector< std::shared_ptr<EventSetupRecordProvider> >& iProviders)
{
   using std::placeholders::_1;
   std::shared_ptr<DependentRecordIntervalFinder> newFinder = make_shared_noexcept_false<DependentRecordIntervalFinder>(key());

   std::shared_ptr<EventSetupRecordIntervalFinder> old = swapFinder(newFinder);

   for(auto const& p: iProviders) { newFinder->addProviderWeAreDependentOn(p); };
   //if a finder was already set, add it as a depedency.  This is done to ensure that the IOVs properly change even if the
   // old finder does not update each time a dependent record does change
   if(old.get() != nullptr) {
      newFinder->setAlternateFinder(old);
   }
}
void 
EventSetupRecordProvider::usePreferred(const DataToPreferredProviderMap& iMap)
{
  using std::placeholders::_1;
  for_all(providers_, std::bind(&EventSetupRecordProvider::addProxiesToRecordHelper,this,_1,iMap));
  if (1 < multipleFinders_->size()) {
     std::shared_ptr<IntersectingIOVRecordIntervalFinder> intFinder = make_shared_noexcept_false<IntersectingIOVRecordIntervalFinder>(key_);
     intFinder->swapFinders(*multipleFinders_);
     finder_ = intFinder;
  }
  //now we get rid of the temporary
  multipleFinders_.reset(nullptr);
}

void 
EventSetupRecordProvider::addProxiesToRecord(std::shared_ptr<DataProxyProvider> iProvider,
                                const EventSetupRecordProvider::DataToPreferredProviderMap& iMap) {
   typedef DataProxyProvider::KeyedProxies ProxyList ;
   typedef EventSetupRecordProvider::DataToPreferredProviderMap PreferredMap;
   
   const ProxyList& keyedProxies(iProvider->keyedProxies(this->key())) ;
   ProxyList::const_iterator finishedProxyList(keyedProxies.end()) ;
   for (ProxyList::const_iterator keyedProxy(keyedProxies.begin()) ;
        keyedProxy != finishedProxyList ;
        ++keyedProxy) {
      PreferredMap::const_iterator itFound = iMap.find(keyedProxy->first);
      if(iMap.end() != itFound) {
         if( itFound->second.type_ != keyedProxy->second->providerDescription()->type_ ||
            itFound->second.label_ != keyedProxy->second->providerDescription()->label_ ) {
            //this is not the preferred provider
            continue;
         }
      }
      record_.add((*keyedProxy).first , (*keyedProxy).second.get()) ;
   }
}
      
void 
EventSetupRecordProvider::addRecordTo(EventSetupProvider& iEventSetupProvider) {
   record_.set(this->validityInterval());
   iEventSetupProvider.addRecordToEventSetup(record_);
}
      
//
// const member functions
//
void 
EventSetupRecordProvider::resetTransients()
{

   using std::placeholders::_1;
   if(checkResetTransients())  {
      for_all(providers_, std::bind(&DataProxyProvider::resetProxiesIfTransient,_1,key_)); 
   }
}

      
void 
EventSetupRecordProvider::addRecordToIfValid(EventSetupProvider& iEventSetupProvider,
                                          const IOVSyncValue& iTime)
{
   if(setValidityIntervalFor(iTime)) {
      addRecordTo(iEventSetupProvider);
   } 
}

bool 
EventSetupRecordProvider::setValidityIntervalFor(const IOVSyncValue& iTime)
{
   //we want to wait until after the first event of a new run before
   // we reset any transients just in case some modules get their data at beginRun or beginLumi
   // and others wait till the first event
   if(!lastSyncWasBeginOfRun_) {
      resetTransients();
   }
   lastSyncWasBeginOfRun_=iTime.eventID().event() == 0;
   
   if(validityInterval_.validFor(iTime)) {
      return true;
   }
   bool returnValue = false;
   //need to see if we get a new interval
   if(nullptr != finder_.get()) {
      IOVSyncValue oldFirst(validityInterval_.first());
      
      validityInterval_ = finder_->findIntervalFor(key_, iTime);
      //are we in a valid range?
      if(validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue()) {
         returnValue = true;
         //did we actually change?
         if(oldFirst != validityInterval_.first()) {
            //tell all Providers to update
            for(auto& provider : providers_) {
               provider->newInterval(key_, validityInterval_);
            }
            cacheReset();
         }
      }
   }
   return returnValue;
}

void 
EventSetupRecordProvider::resetProxies()
{
   using std::placeholders::_1;
   cacheReset();
   for_all(providers_, std::bind(&DataProxyProvider::resetProxies,_1,key_));
   //some proxies only clear if they were accessed transiently,
   // since resetProxies resets that flag, calling resetTransients
   // will force a clear
   for_all(providers_, std::bind(&DataProxyProvider::resetProxiesIfTransient,_1,key_)); 

}

void
EventSetupRecordProvider::getReferencedESProducers(std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> >& referencedESProducers) {
   record().getESProducers(referencedESProducers[key_]);
}

void
EventSetupRecordProvider::fillReferencedDataKeys(std::map<DataKey, ComponentDescription const*>& referencedDataKeys) {
   record().fillReferencedDataKeys(referencedDataKeys);
}

void
EventSetupRecordProvider::resetRecordToProxyPointers(DataToPreferredProviderMap const& iMap) {
   using std::placeholders::_1;
   record().clearProxies();
   for_all(providers_, std::bind(&EventSetupRecordProvider::addProxiesToRecordHelper, this, _1, iMap));
}

void 
EventSetupRecordProvider::cacheReset()
{
   record().cacheReset();
}

bool 
EventSetupRecordProvider::checkResetTransients() 
{
   return record().transientReset();
}
      

std::set<EventSetupRecordKey> 
EventSetupRecordProvider::dependentRecords() const
{
  return dependencies(key());
}

std::set<ComponentDescription> 
EventSetupRecordProvider::proxyProviderDescriptions() const
{
   using std::placeholders::_1;
   std::set<ComponentDescription> descriptions;
   std::transform(providers_.begin(), providers_.end(),
                  std::inserter(descriptions,descriptions.end()),
                  std::bind(&DataProxyProvider::description,_1));
   return descriptions;
}

std::shared_ptr<DataProxyProvider> 
EventSetupRecordProvider::proxyProvider(ComponentDescription const& iDesc) {
   using std::placeholders::_1;
   auto itFound = std::find_if(providers_.begin(),providers_.end(),
                std::bind(std::equal_to<ComponentDescription>(), 
                            iDesc, 
                            std::bind(&DataProxyProvider::description,_1)));
   if(itFound == providers_.end()){
      return std::shared_ptr<DataProxyProvider>();
   }
   return get_underlying_safe(*itFound);
}

std::shared_ptr<DataProxyProvider> 
EventSetupRecordProvider::proxyProvider(ParameterSetIDHolder const& psetID) {
   for (auto& dataProxyProvider : providers_) {
      if (dataProxyProvider->description().pid_ == psetID.psetID()) {
         return get_underlying_safe(dataProxyProvider);
      }
   }
   return std::shared_ptr<DataProxyProvider>();
}

void
EventSetupRecordProvider::resetProxyProvider(ParameterSetIDHolder const& psetID, std::shared_ptr<DataProxyProvider> const& sharedDataProxyProvider) {
   for (auto& dataProxyProvider : providers_) {
      if (dataProxyProvider->description().pid_ == psetID.psetID()) {
         dataProxyProvider = sharedDataProxyProvider;
      }
   }
}

//
// static member functions
//
   }
}
