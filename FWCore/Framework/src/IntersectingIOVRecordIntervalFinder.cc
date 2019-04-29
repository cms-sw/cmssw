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

#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"

namespace edm {
   namespace eventsetup {

IntersectingIOVRecordIntervalFinder::IntersectingIOVRecordIntervalFinder(const EventSetupRecordKey& iKey)
{
   findingRecordWithKey(iKey);
}

IntersectingIOVRecordIntervalFinder::~IntersectingIOVRecordIntervalFinder()
{
}

void 
IntersectingIOVRecordIntervalFinder::swapFinders(std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>& iFinders)
{
   finders_.swap(iFinders);
}

bool IntersectingIOVRecordIntervalFinder::hasLegacyESSource() const {
   for (auto const& finder : finders_) {
      if (finder->legacyESSource()) {
         return true;
      }
   }
   return false;
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

   for(auto& finder : finders_) {
      ValidityInterval test = finder->findIntervalFor(iKey, iTime);
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

void IntersectingIOVRecordIntervalFinder::doResetInterval(const eventsetup::EventSetupRecordKey& key) {
   for (auto& finder : finders_) {
      finder->resetInterval(key);
   }
}

bool IntersectingIOVRecordIntervalFinder::isLegacyESSource() const {
   return false;
}

bool IntersectingIOVRecordIntervalFinder::isLegacyOutOfValidityInterval(const EventSetupRecordKey& iKey,
                                                                        const IOVSyncValue& iTime) const {
   for (auto const& finder : finders_) {
      if (finder->legacyOutOfValidityInterval(iKey, iTime)) {
         return true;
      }
   }
   return false;
}

   }
}
