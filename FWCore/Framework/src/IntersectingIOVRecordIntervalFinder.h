#ifndef FWCore_Framework_IntersectingIOVRecordIntervalFinder_h
#define FWCore_Framework_IntersectingIOVRecordIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     IntersectingIOVRecordIntervalFinder
// 
/**\class IntersectingIOVRecordIntervalFinder IntersectingIOVRecordIntervalFinder.h FWCore/Framework/interface/IntersectingIOVRecordIntervalFinder.h

 Description: A RecordIntervalFinder which determines IOVs by taking the intersection of IOVs of other RecordIntervalFinders

 Usage:
    Used internally by the framework

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Aug 19 13:20:34 EDT 2008
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
   namespace eventsetup {

      class IntersectingIOVRecordIntervalFinder : public EventSetupRecordIntervalFinder {
         
      public:
         explicit IntersectingIOVRecordIntervalFinder(const EventSetupRecordKey&);
         ~IntersectingIOVRecordIntervalFinder() override;
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         void swapFinders(std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>&);
      protected:
         void setIntervalFor(const EventSetupRecordKey&,
                                     const IOVSyncValue& , 
                                     ValidityInterval&) override;
         
      private:
         IntersectingIOVRecordIntervalFinder(const IntersectingIOVRecordIntervalFinder&) = delete; // stop default
         
         const IntersectingIOVRecordIntervalFinder& operator=(const IntersectingIOVRecordIntervalFinder&) = delete; // stop default
         
         // ---------- member data --------------------------------
         std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>> finders_;
      };
   }
}

#endif
