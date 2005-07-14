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

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"


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
      EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey) : key_(iKey)
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
   assert(providers_.end() == std::find(providers_.begin(),
                                          providers_.end(),
                                          iProvider));
   
   providers_.push_back(iProvider);
   //do it now, in future may delay this till later
   addProxiesToRecord(iProvider);
}

void 
EventSetupRecordProvider::addFinder(boost::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   finder_ = iFinder;
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

//
// const member functions
//

void 
EventSetupRecordProvider::addRecordToIfValid(EventSetupProvider& iEventSetupProvider,
                                          const Timestamp& iTime)
{
   if(setValidityIntervalFor(iTime)) {
      addRecordTo(iEventSetupProvider);
   } 
}

bool 
EventSetupRecordProvider::setValidityIntervalFor(const Timestamp& iTime)
{
   if(validityInterval_.validFor(iTime)) {
      return true;
   }
   bool returnValue = false;
   //need to see if we get a new interval
   if(0 != finder_.get()) {
      Timestamp oldFirst(validityInterval_.first());
      
      validityInterval_ = finder_->findIntervalFor(key_, iTime);
      //are we in a valid range?
      if(validityInterval_.first() != Timestamp::invalidTimestamp()) {
         returnValue = true;
         //did we actually change?
         if(oldFirst != validityInterval_.first()) {
            //tell all Providers to update
            for(std::vector<boost::shared_ptr<DataProxyProvider> >::iterator itProvider = providers_.begin();
                itProvider != providers_.end();
                ++itProvider) {
               (*itProvider)->newInterval(key_, validityInterval_);
            }
         }
      }
   }
   return returnValue;
}


std::set<EventSetupRecordKey> 
EventSetupRecordProvider::dependentRecords() const
{
   return std::set<EventSetupRecordKey>();
}

//
// static member functions
//
   }
}
