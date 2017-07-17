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
#include <vector>
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

// user include files

// forward declarations
namespace edm {

  class EventRange {

   public:

      EventRange();

      EventRange(RunNumber_t startRun, LuminosityBlockNumber_t startLumi, EventNumber_t startEvent,
                 RunNumber_t endRun, LuminosityBlockNumber_t endLumi, EventNumber_t endEvent);

      EventRange(EventID const& begin, EventID const& end);
//      virtual ~EventRange();

      // ---------- const member functions ---------------------
      EventID     startEventID() const {return startEventID_; }
      EventID       endEventID() const {return   endEventID_; }
      RunNumber_t     startRun() const {return    startEventID_.run(); }
      RunNumber_t       endRun() const {return      endEventID_.run(); }
      LuminosityBlockNumber_t   startLumi() const {return startEventID_.luminosityBlock(); }
      LuminosityBlockNumber_t     endLumi() const {return   endEventID_.luminosityBlock(); }
      EventNumber_t startEvent() const {return  startEventID_.event(); }
      EventNumber_t   endEvent() const {return    endEventID_.event(); }

   private:

      // ---------- member data --------------------------------
      //RunNumber_t   startRun_;
      //RunNumber_t   endRun_;
      //LuminosityBlockNumber_t   startLumi_;
      //LuminosityBlockNumber_t   endLumi_;
      //EventNumber_t startEvent_;
      //EventNumber_t endEvent_;
      EventID startEventID_;
      EventID endEventID_;
  };

  std::ostream& operator<<(std::ostream& oStream, EventRange const& iID);
  bool contains(EventRange const& lh, EventID const& rh);
  bool contains_(EventRange const& lh, EventID const& rh);
  bool contains(EventRange const& lh, EventRange const& rh);
  bool lessThan(EventRange const& lh, EventRange const& rh);
  bool lessThanSpecial(EventRange const& lh, EventRange const& rh);
  bool overlaps(EventRange const& lh, EventRange const& rh);
  bool distinct(EventRange const& lh, EventRange const& rh);
  std::vector<EventRange>& sortAndRemoveOverlaps(std::vector<EventRange>& eventRange);

}
#endif
