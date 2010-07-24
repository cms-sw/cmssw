#ifndef DataFormats_Provenance_EventRange_h
#define DataFormats_Provenance_EventRange_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     EventRange
//

/*
 Description: Holds run and event range.

 Usage:
    <usage>

*/
//
//

// system include files
#include <functional>
#include <iosfwd>
#include "DataFormats/Provenance/interface/MinimalEventID.h"

// user include files

// forward declarations
namespace edm {

   typedef unsigned int EventNumber_t;
   typedef unsigned int RunNumber_t;
   
  class EventRange {

   public:

      EventRange() {
        edm::EventRange(0,0,0,0);
      }

      EventRange(RunNumber_t startRun, EventNumber_t startEvent,
                           RunNumber_t endRun,   EventNumber_t endEvent) :
	startRun_(startRun), endRun_(endRun),  startEvent_(startEvent), endEvent_(endEvent){
         // Special cases since 0 means maximum
         edm::MinimalEventID dummy = edm::MinimalEventID();
         if (startEvent == 0) {
           startEvent_ = dummy.maxEventNumber();
         }
         if (endEvent == 0) {
           endEvent_ = dummy.maxEventNumber();
         }
         startEventID_ = edm::MinimalEventID(startRun_, startEvent_);
         endEventID_   = edm::MinimalEventID(endRun_,  endEvent_);
      }

//      virtual ~EventRange();

      // ---------- const member functions ---------------------
      MinimalEventID     startEventID() const {return startEventID_; }
      MinimalEventID       endEventID() const {return   endEventID_; }
      RunNumber_t     startRun() const {return     startRun_; }
      RunNumber_t       endRun() const {return       endRun_; }
      EventNumber_t startEvent() const {return   startEvent_; }
      EventNumber_t   endEvent() const {return     endEvent_; }

   private:

      // ---------- member data --------------------------------
      RunNumber_t   startRun_;
      RunNumber_t   endRun_;
      EventNumber_t startEvent_;
      EventNumber_t endEvent_;
      MinimalEventID startEventID_;
      MinimalEventID endEventID_;
  };

  std::ostream& operator<<(std::ostream& oStream, EventRange const& iID);
  bool contains(EventRange const& lh, MinimalEventID const& rh);
  bool contains(EventRange const& lh, EventRange const& rh);
  bool overlaps(EventRange const& lh, EventRange const& rh);
  bool distinct(EventRange const& lh, EventRange const& rh);

}
#endif
