#ifndef Common_EventID_h
#define Common_EventID_h
// -*- C++ -*-
//
// Package:     EDProduct
// Class  :     EventID
// 
/**\class EventID EventID.h DataFormats/Common/interface/EventID.h

 Description: Holds run and event number

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Aug  8 15:13:14 EDT 2005
// $Id: EventID.h,v 1.3 2005/09/01 23:30:48 wmtan Exp $
//

// system include files
#include <iostream>

// user include files

// forward declarations
namespace edm {

   typedef unsigned long RunNumber_t;
   typedef unsigned long EventNumber_t;

   
class EventID
{

   public:
   
   
      EventID() : run_(0), event_(0) {}
      EventID(RunNumber_t iRun, EventNumber_t iEvent) :run_(iRun), event_(iEvent) {}
      
      //FIXME: only used for backwards compatibility
      EventID(EventNumber_t iEvent) : run_(1UL), event_(iEvent) {} 
      //virtual ~EventID();

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
      
      bool operator==(const EventID& iRHS) const {
         return iRHS.run_ == run_ && iRHS.event_ == event_;
      }
      bool operator!=(const EventID& iRHS) const {
         return ! (*this == iRHS);
      }
      
      bool operator<(const EventID& iRHS) const {
         return doOp<std::less>(iRHS);
      }
      bool operator<=(const EventID& iRHS) const {
         return doOp<std::less_equal>(iRHS);
      }
      bool operator>(const EventID& iRHS) const {
         return doOp<std::greater>(iRHS);
      }
      bool operator>=(const EventID& iRHS) const {
         return doOp<std::greater_equal>(iRHS);
      }
      // ---------- static functions ---------------------------

      static EventNumber_t maxEventNumber() {
         return 0xFFFFFFFFUL;
      }
   
      static EventID firstValidEvent() {
         return EventID(1, 1);
      }
      // ---------- member functions ---------------------------
   
   private:
      template< template <typename> class Op >
      bool doOp(const EventID& iRHS) const {
         //Run takes presidence for comparisions
         if(run_ == iRHS.run_) {
            Op<EventNumber_t> op_e;
            return op_e(event_, iRHS.event_);
         }
         Op<RunNumber_t> op;
         return op(run_, iRHS.run_) ;
      }
      //EventID(const EventID&); // stop default

      //const EventID& operator=(const EventID&); // stop default

      // ---------- member data --------------------------------
      RunNumber_t run_;
      EventNumber_t event_;
};

inline
std::ostream& operator<<(std::ostream& oStream, const EventID& iID) {
   oStream<< "run: "<< iID.run()<<" event: "<<iID.event();
   return oStream;
}

}
#endif
