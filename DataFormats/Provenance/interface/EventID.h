#ifndef DataFormats_Provenance_EventID_h
#define DataFormats_Provenance_EventID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     EventID
//
/**\class edm::EventID

 Description: Holds run, lumi, and event numbers.

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Aug  8 15:13:14 EDT 2005
//

// system include files
#include <iosfwd>

// user include files
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

// forward declarations
namespace edm {

  class EventID {

    public:
      EventID() : run_(invalidRunNumber), luminosityBlock_(invalidLuminosityBlockNumber), event_(invalidEventNumber) {}
      EventID(RunNumber_t iRun, LuminosityBlockNumber_t iLumi, EventNumber_t iEvent) :
        run_(iRun), luminosityBlock_(iLumi), event_(iEvent) {}

      // ---------- const member functions ---------------------
      RunNumber_t run() const { return run_; }
      LuminosityBlockNumber_t luminosityBlock() const { return luminosityBlock_; }
      EventNumber_t event() const { return event_; }

      //moving from one EventID to another one
      EventID next(LuminosityBlockNumber_t const& lumi) const {
         if(event_ != maxEventNumber()) {
            return EventID(run_, lumi, event_ + 1);
         }
         return EventID(run_ + 1, lumi, 1);
      }
      EventID nextRun(LuminosityBlockNumber_t const& lumi) const {
         return EventID(run_ + 1, lumi, 0);
      }
      EventID nextRunFirstEvent(LuminosityBlockNumber_t const& lumi) const {
         return EventID(run_ + 1, lumi, 1);
      }
      EventID previousRunLastEvent(LuminosityBlockNumber_t const& lumi) const {
         if(run_ > 1) {
            return EventID(run_ - 1, lumi, maxEventNumber());
         }
         return EventID();
      }

      EventID previous(LuminosityBlockNumber_t const& lumi) const {
         if(event_ > 1) {
            return EventID(run_, lumi, event_-1);
         }
         if(run_ != 0) {
            return EventID(run_ - 1, lumi, maxEventNumber());
         }
         return EventID();
      }

      bool operator<(EventID const& iRHS) const {
        if (run_ < iRHS.run_) return true;
        if (run_ > iRHS.run_) return false;
        if (luminosityBlock_ < iRHS.luminosityBlock_) return true;
        if (luminosityBlock_ > iRHS.luminosityBlock_) return false;
        return (event_ < iRHS.event_);
      }

      bool operator>=(EventID const& iRHS) const {
         return !(*this < iRHS);
      }

      bool operator==(EventID const& iRHS) const {
         return !(*this < iRHS || iRHS < *this);
      }

      bool operator!=(EventID const& iRHS) const {
         return !(*this == iRHS);
      }

      bool operator<=(EventID const& iRHS) const {
         return (*this < iRHS || *this == iRHS);
      }

      bool operator>(EventID const& iRHS) const {
         return !(*this <= iRHS);
      }

      // ---------- static functions ---------------------------

      static RunNumber_t maxRunNumber() {
         return 0xFFFFFFFFU;
      }

      static LuminosityBlockNumber_t maxLuminosityBlockNumber() {
         return 0xFFFFFFFFU;
      }

      static EventNumber_t maxEventNumber() {
         return 0xFFFFFFFFFFFFFFFFULL;
      }

      static EventID firstValidEvent() {
         return EventID(1, 1, 1);
      }
      // ---------- member functions ---------------------------

      void setLuminosityBlockNumber(LuminosityBlockNumber_t const& lb) {
        luminosityBlock_ = lb;
      }
   private:
      //EventID(EventID const&); // stop default

      //EventID const& operator=(EventID const&); // stop default

      // ---------- member data --------------------------------
      RunNumber_t run_;
      LuminosityBlockNumber_t luminosityBlock_;
      EventNumber_t event_;
  };

  std::ostream& operator<<(std::ostream& oStream, EventID const& iID);

  inline
  EventID const& min(EventID const& lh, EventID const& rh) {
    return (rh < lh ? rh : lh);
  }

  inline
  EventID const& max(EventID const& lh, EventID const& rh) {
    return (rh < lh ? lh : rh);
  }
}
#endif
