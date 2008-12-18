#ifndef DataFormats_Provenance_EventID_h
#define DataFormats_Provenance_EventID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     EventID
// 
/**\class EventID EventID.h DataFormats/Provenance/interface/EventID.h

 Description: Holds run and event number.

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

// forward declarations
namespace edm {

   typedef unsigned int EventNumber_t;

   
class EventID {

   public:
   
   
      EventID() : run_(0), event_(0) {}
      EventID(RunNumber_t iRun, EventNumber_t iEvent) :
	run_(iRun), event_(iEvent) {}
      
      // ---------- const member functions ---------------------
      RunNumber_t run() const { return run_; }
      EventNumber_t event() const { return event_; }
   
      //moving from one EventID to another one
      EventID next() const {
         if(event_ != maxEventNumber()) {
            return EventID(run_, event_+1);
         }
         return EventID(run_+1, 1);
      }
      EventID nextRun() const {
         return EventID(run_+1, 0);
      }
      EventID nextRunFirstEvent() const {
         return EventID(run_+1, 1);
      }
      EventID previousRunLastEvent() const {
         if(run_ > 1) {
            return EventID(run_-1, maxEventNumber());
         }
         return EventID(0,0);
      }
   
      EventID previous() const {
         if(event_ > 1) {
            return EventID(run_, event_-1);
         }
         if(run_ != 0) {
            return EventID(run_ -1, maxEventNumber());
         }
         return EventID(0,0);
      }
      
      bool operator<(EventID const& iRHS) const {
         return (run_ == iRHS.run_ ? event_ < iRHS.event_ : run_ < iRHS.run_);
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

      static EventNumber_t maxEventNumber() {
         return 0xFFFFFFFFU;
      }
   
      static EventID firstValidEvent() {
         return EventID(1, 1);
      }
      // ---------- member functions ---------------------------
   
   private:
      //EventID(EventID const&); // stop default

      //EventID const& operator=(EventID const&); // stop default

      // ---------- member data --------------------------------
      RunNumber_t run_;
      EventNumber_t event_;
};

std::ostream& operator<<(std::ostream& oStream, EventID const& iID);

}
#endif
