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
#include "DataFormats/Provenance/interface/EventID.h"

// user include files

// forward declarations
namespace edm {

   typedef unsigned int EventNumber_t;
   typedef unsigned int LumiNumber_t;
   typedef unsigned int RunNumber_t;

  class EventRange {

   public:

      EventRange();

      EventRange(RunNumber_t startRun, LumiNumber_t startLumi, EventNumber_t startEvent,
                 RunNumber_t endRun, LumiNumber_t endLumi, EventNumber_t endEvent);
//      virtual ~EventRange();

      // ---------- const member functions ---------------------
      EventID     startEventID() const {return startEventID_; }
      EventID       endEventID() const {return   endEventID_; }
      RunNumber_t     startRun() const {return    startEventID_.run(); }
      RunNumber_t       endRun() const {return      endEventID_.run(); }
      LumiNumber_t   startLumi() const {return startEventID_.luminosityBlock(); }
      LumiNumber_t     endLumi() const {return   endEventID_.luminosityBlock(); }
      EventNumber_t startEvent() const {return  startEventID_.event(); }
      EventNumber_t   endEvent() const {return    endEventID_.event(); }

   private:

      // ---------- member data --------------------------------
      //RunNumber_t   startRun_;
      //RunNumber_t   endRun_;
      //LumiNumber_t   startLumi_;
      //LumiNumber_t   endLumi_;
      //EventNumber_t startEvent_;
      //EventNumber_t endEvent_;
      EventID startEventID_;
      EventID endEventID_;
  };

  std::ostream& operator<<(std::ostream& oStream, EventRange const& iID);
  bool contains(EventRange const& lh, EventID const& rh);
  bool contains_(EventRange const& lh, EventID const& rh);
  bool contains(EventRange const& lh, EventRange const& rh);
  bool overlaps(EventRange const& lh, EventRange const& rh);
  bool distinct(EventRange const& lh, EventRange const& rh);

}
#endif
