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
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Utilities/interface/Algorithms.h"


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
     multipleFinders_(new std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >()) 
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
EventSetupRecordProvider::setDependentProviders(const std::vector< boost::shared_ptr<EventSetupRecordProvider> >&)
{
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

//
// const member functions
//

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
