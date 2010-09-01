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

      EventRange(): 
        startRun_(0), endRun_(0),  
        // Special cases since 0 means maximum
        startEvent_(edm::MinimalEventID::maxEventNumber()),
        endEvent_(edm::MinimalEventID::maxEventNumber()),
        startEventID_(startRun_, startEvent_),
        endEventID_(endRun_,  endEvent_)
      {}
     

      EventRange(RunNumber_t startRun, EventNumber_t startEvent,
                           RunNumber_t endRun,   EventNumber_t endEvent) :
        startRun_(startRun), endRun_(endRun),  
        // Special cases since 0 means maximum
        startEvent_(startEvent !=0? startEvent : edm::MinimalEventID::maxEventNumber()),
        endEvent_(endEvent !=0? endEvent : edm::MinimalEventID::maxEventNumber()),
        startEventID_(startRun_, startEvent_),
        endEventID_(endRun_,  endEvent_)
      {}

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
