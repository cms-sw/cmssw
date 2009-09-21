#ifndef DataFormats_Provenance_MinimalEventID_h
#define DataFormats_Provenance_MinimalEventID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  a     
//

/*
 Description: Holds run and event number only. No luminosity block number is specified.

 Usage:
    <usage>

*/
//
//

// system include files
#include <functional>
#include <iosfwd>
#include "boost/cstdint.hpp"

// user include files

// forward declarations
namespace edm {

   typedef unsigned int EventNumber_t;
   typedef unsigned int RunNumber_t;
   
  class MinimalEventID {

    public:
   
      MinimalEventID() : run_(0), event_(0) {}

      MinimalEventID(RunNumber_t iRun, EventNumber_t iEvent) :
	run_(iRun), event_(iEvent) {}
      
      // ---------- const member functions ---------------------
      RunNumber_t run() const { return run_; }
      EventNumber_t event() const { return event_; }
   
      bool operator<(MinimalEventID const& iRHS) const {
         return (run_ == iRHS.run_ ? event_ < iRHS.event_ : run_ < iRHS.run_);
      }

      bool operator>=(MinimalEventID const& iRHS) const {
         return !(*this < iRHS);
      }

      bool operator==(MinimalEventID const& iRHS) const {
         return !(*this < iRHS || iRHS < *this);
      }

      bool operator!=(MinimalEventID const& iRHS) const {
         return !(*this == iRHS);
      }

      bool operator<=(MinimalEventID const& iRHS) const {
         return (*this < iRHS || *this == iRHS);
      }

      bool operator>(MinimalEventID const& iRHS) const {
         return !(*this <= iRHS);
      }

      // ---------- static functions ---------------------------

      static EventNumber_t maxEventNumber() {
         return 0xFFFFFFFFU;
      }
   
      static MinimalEventID firstValidEvent() {
         return MinimalEventID(1, 1);
      }
      // ---------- member functions ---------------------------
   
   private:

      // ---------- member data --------------------------------
      RunNumber_t run_;
      EventNumber_t event_;
  };

  std::ostream& operator<<(std::ostream& oStream, MinimalEventID const& iID);

}
#endif
