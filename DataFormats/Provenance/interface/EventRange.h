#ifndef DataFormats_Provenance_EventRange_h
#define DataFormats_Provenance_EventRange_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     LuminosityBlockRange
//
/**\class EventRange LuminosityBlockRange.h DataFormats/Provenance/interface/EventRange.h

 Description: Holds run and event range.

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
#include "DataFormats/Provenance/interface/EventID.h"

// forward declarations
namespace edm {

//   typedef unsigned int LuminosityBlockNumber_t;


class EventRange
{

   public:

      EventRange() {
        edm::EventRange(0,0,0,0);
      }

      EventRange(RunNumber_t startRun, EventNumber_t startEvent,
                           RunNumber_t endRun,   EventNumber_t endEvent) :
	startRun_(startRun), endRun_(endRun),  startEvent_(startEvent), endEvent_(endEvent){
         // Special cases since 0 means maximum
         edm::EventID dummy = edm::EventID();
         if (startEvent == 0) {
           startEvent_ = dummy.maxEventNumber();
         }
         if (endEvent == 0) {
           endEvent_ = dummy.maxEventNumber();
         }
         startEventID_ = edm::EventID(startRun_,startEvent_);
         endEventID_   = edm::EventID(endRun_,  endEvent_);
      }

//      virtual ~EventRange();

      // ---------- const member functions ---------------------
      EventID     startEventID() const {return startEventID_; }
      EventID       endEventID() const {return   endEventID_; }
      RunNumber_t     startRun() const {return     startRun_; }
      RunNumber_t       endRun() const {return       endRun_; }
      EventNumber_t startEvent() const {return   startEvent_; }
      EventNumber_t   endEvent() const {return     endEvent_; }

      bool contains(EventID const& test) const;


   private:

      // ---------- member data --------------------------------
      RunNumber_t   startRun_,    endRun_;
      EventNumber_t startEvent_,  endEvent_;
      EventID       startEventID_,endEventID_;
};

std::ostream& operator<<(std::ostream& oStream, EventRange const& iID);

}
#endif
