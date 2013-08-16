// -*- C++ -*-
//
// Package:     Framework
// Class  :     IntersectingIOVRecordIntervalFinder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Aug 19 13:20:41 EDT 2008
//

// system include files

// user include files
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
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
IntersectingIOVRecordIntervalFinder::IntersectingIOVRecordIntervalFinder(const EventSetupRecordKey& iKey)
{
   findingRecordWithKey(iKey);
}

// IntersectingIOVRecordIntervalFinder::IntersectingIOVRecordIntervalFinder(const IntersectingIOVRecordIntervalFinder& rhs)
// {
//    // do actual copying here;
// }

IntersectingIOVRecordIntervalFinder::~IntersectingIOVRecordIntervalFinder()
{
}

//
// assignment operators
//
// const IntersectingIOVRecordIntervalFinder& IntersectingIOVRecordIntervalFinder::operator=(const IntersectingIOVRecordIntervalFinder& rhs)
// {
//   //An exception safe implementation is
//   IntersectingIOVRecordIntervalFinder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
IntersectingIOVRecordIntervalFinder::swapFinders(std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >& iFinders)
{
   finders_.swap(iFinders);
}

void 
IntersectingIOVRecordIntervalFinder::setIntervalFor(const EventSetupRecordKey& iKey,
                                                    const IOVSyncValue& iTime, 
                                                    ValidityInterval& oInterval)
{
   if(finders_.empty()) {
      oInterval = ValidityInterval::invalidInterval();
      return;
   }
   
   bool haveAValidRecord = false;
   bool haveUnknownEnding = false;
   ValidityInterval newInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());

   for(std::vector<boost::shared_ptr<EventSetupRecordIntervalFinder> >::iterator it = finders_.begin(),
       itEnd = finders_.end(); it != itEnd; ++it) {
      ValidityInterval test = (*it)->findIntervalFor(iKey, iTime);
      if ( test != ValidityInterval::invalidInterval() ) {
         haveAValidRecord =true;
         if(newInterval.first() < test.first()) {
            newInterval.setFirst(test.first());
         }
         if(newInterval.last() > test.last()) {
            newInterval.setLast(test.last());
         }
         if(test.last() == IOVSyncValue::invalidIOVSyncValue()) {
            haveUnknownEnding=true;
         }
      } else {
         //if it is invalid then we must check on each new IOVSyncValue so that 
         // we can find the time when it is valid
         haveUnknownEnding=true;
      }
   }
   
   if(!haveAValidRecord) {
      //If no Finder has a valid time, then this record is also invalid for this time
      newInterval = ValidityInterval::invalidInterval();
   } else if(haveUnknownEnding) {
      newInterval.setLast(IOVSyncValue::invalidIOVSyncValue());
   }
   oInterval = newInterval;
}

//
// const member functions
//

//
// static member functions
//
   }
}
