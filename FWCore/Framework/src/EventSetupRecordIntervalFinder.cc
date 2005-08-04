// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIntervalFinder
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed Mar 30 14:27:26 EST 2005
// $Id: EventSetupRecordIntervalFinder.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"


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
//EventSetupRecordIntervalFinder::EventSetupRecordIntervalFinder()
//{
//}

// EventSetupRecordIntervalFinder::EventSetupRecordIntervalFinder(const EventSetupRecordIntervalFinder& rhs)
// {
//    // do actual copying here;
// }

EventSetupRecordIntervalFinder::~EventSetupRecordIntervalFinder()
{
}

//
// assignment operators
//
// const EventSetupRecordIntervalFinder& EventSetupRecordIntervalFinder::operator=(const EventSetupRecordIntervalFinder& rhs)
// {
//   //An exception safe implementation is
//   EventSetupRecordIntervalFinder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
const ValidityInterval& 
EventSetupRecordIntervalFinder::findIntervalFor(const EventSetupRecordKey& iKey,
                                              const IOVSyncValue& iInstance)
{
   Intervals::iterator itFound = intervals_.find(iKey);
   assert(itFound != intervals_.end()) ;
   if(! itFound->second.validFor(iInstance)) {
      setIntervalFor(iKey, iInstance, itFound->second);
   }
   return itFound->second;
}

void 
EventSetupRecordIntervalFinder::findingRecordWithKey(const EventSetupRecordKey& iKey) {
   intervals_.insert(Intervals::value_type(iKey, ValidityInterval()));
}

//
// const member functions
//
std::set<EventSetupRecordKey> 
EventSetupRecordIntervalFinder::findingForRecords() const
{
   std::set<EventSetupRecordKey> returnValue;
   
   Intervals::const_iterator itEnd = intervals_.end();
   for(Intervals::const_iterator itEntry = intervals_.begin();
       itEntry != itEnd;
       ++itEntry) {
      returnValue.insert(returnValue.end(), itEntry->first);
   }
   return returnValue;
}

//
// static member functions
//
   }
}
