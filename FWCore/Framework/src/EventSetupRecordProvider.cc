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
#include "boost/bind.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"




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
EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey) : key_(iKey),
    validityInterval_(), finder_(), providers_(),
    multipleFinders_(new std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >()),
    lastSyncWasBeginOfRun_(true)
{
}

// EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordProvider& rhs)
// {
//    // do actual copying here;
// }

EventSetupRecordProvider::~EventSetupRecordProvider()
{
}

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
EventSetupRecordProvider::add(boost::shared_ptr<DataProxyProvider> iProvider)
{
   assert(iProvider->isUsingRecord(key_));
   assert(!search_all(providers_, iProvider));
   providers_.push_back(iProvider);
}

void 
EventSetupRecordProvider::addFinder(boost::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   boost::shared_ptr<EventSetupRecordIntervalFinder> oldFinder = finder_;  
   finder_ = iFinder;
   if (0 != multipleFinders_.get()) {
     multipleFinders_->push_back(iFinder);
   } else {
     //dependent records set there finders after the multipleFinders_ has been released
     // but they also have never had a finder set
     if(0 != oldFinder.get()) {
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
EventSetupRecordProvider::setDependentProviders(const std::vector< boost::shared_ptr<EventSetupRecordProvider> >& iProviders)
{
   boost::shared_ptr< DependentRecordIntervalFinder > newFinder(
                                                                new DependentRecordIntervalFinder(key()));
   
   boost::shared_ptr<EventSetupRecordIntervalFinder> old = swapFinder(newFinder);
   for_all(iProviders, boost::bind(std::mem_fun(&DependentRecordIntervalFinder::addProviderWeAreDependentOn), &(*newFinder), _1));
   //if a finder was already set, add it as a depedency.  This is done to ensure that the IOVs properly change even if the
   // old finder does not update each time a dependent record does change
   if(old.get() != 0) {
      newFinder->setAlternateFinder(old);
   }
}
void 
EventSetupRecordProvider::usePreferred(const DataToPreferredProviderMap& iMap)
{
  for_all(providers_, boost::bind(&EventSetupRecordProvider::addProxiesToRecord,this,_1,iMap));
  if (1 < multipleFinders_->size()) {
     
     boost::shared_ptr<IntersectingIOVRecordIntervalFinder> intFinder(new IntersectingIOVRecordIntervalFinder(key_));
     intFinder->swapFinders(*multipleFinders_);
     finder_ = intFinder;
  }
  //now we get rid of the temporary
  multipleFinders_.reset(0);
}

void 
EventSetupRecordProvider::addProxiesToRecord(boost::shared_ptr<DataProxyProvider> iProvider,
                                const EventSetupRecordProvider::DataToPreferredProviderMap& iMap) {
   typedef DataProxyProvider::KeyedProxies ProxyList ;
   typedef EventSetupRecordProvider::DataToPreferredProviderMap PreferredMap;
   
   EventSetupRecord& rec = record();
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
      rec.add((*keyedProxy).first , (*keyedProxy).second.get()) ;
   }
}
      
void 
EventSetupRecordProvider::addRecordTo(EventSetupProvider& iEventSetupProvider) {
   EventSetupRecord& rec = record();
   rec.set(this->validityInterval());
   iEventSetupProvider.addRecordToEventSetup(rec);
}
      
//
// const member functions
//
void 
EventSetupRecordProvider::resetTransients()
{
   if(checkResetTransients()) {
      for_all(providers_, boost::bind(&DataProxyProvider::resetProxiesIfTransient,_1,key_)); 
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
   if(0 != finder_.get()) {
      IOVSyncValue oldFirst(validityInterval_.first());
      
      validityInterval_ = finder_->findIntervalFor(key_, iTime);
      //are we in a valid range?
      if(validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue()) {
         returnValue = true;
         //did we actually change?
         if(oldFirst != validityInterval_.first()) {
            //tell all Providers to update
            for(std::vector<boost::shared_ptr<DataProxyProvider> >::iterator itProvider = providers_.begin(),
	        itProviderEnd = providers_.end();
                itProvider != itProviderEnd;
                ++itProvider) {
               (*itProvider)->newInterval(key_, validityInterval_);
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
  cacheReset();
  for_all(providers_, boost::bind(&DataProxyProvider::resetProxies,_1,key_));
  //some proxies only clear if they were accessed transiently,
  // since resetProxies resets that flag, calling resetTransients
  // will force a clear
  for_all(providers_, boost::bind(&DataProxyProvider::resetProxiesIfTransient,_1,key_)); 

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
   return std::set<EventSetupRecordKey>();
}

std::set<ComponentDescription> 
EventSetupRecordProvider::proxyProviderDescriptions() const
{
   std::set<ComponentDescription> descriptions;
   std::transform(providers_.begin(), providers_.end(),
                  std::inserter(descriptions,descriptions.end()),
                  boost::bind(&DataProxyProvider::description,_1));
   return descriptions;
}

boost::shared_ptr<DataProxyProvider> 
EventSetupRecordProvider::proxyProvider(const ComponentDescription& iDesc) const {
   std::vector<boost::shared_ptr<DataProxyProvider> >::const_iterator itFound =
   std::find_if(providers_.begin(),providers_.end(),
                boost::bind(std::equal_to<ComponentDescription>(), 
                            iDesc, 
                            boost::bind(&DataProxyProvider::description,_1)));
   if(itFound == providers_.end()){
      return boost::shared_ptr<DataProxyProvider>();
   }
   return *itFound;
}

      

//
// static member functions
//
   }
}
