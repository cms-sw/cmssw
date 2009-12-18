#ifndef Framework_IOVSyncValue_h
#define Framework_IOVSyncValue_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     IOVSyncValue
// 
/**\class IOVSyncValue IOVSyncValue.h FWCore/Framework/interface/IOVSyncValue.h

 Description: Provides the information needed to synchronize the EventSetup IOV with an Event

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  3 18:35:24 EDT 2005
//

// system include files
#include <functional>

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// forward declarations

namespace edm {
class IOVSyncValue
{

   public:
      IOVSyncValue();
      //virtual ~IOVSyncValue();
      explicit IOVSyncValue(const EventID& iID);
      explicit IOVSyncValue(const Timestamp& iTime);
      IOVSyncValue(const EventID& iID, const Timestamp& iTime);

      // ---------- const member functions ---------------------
      const EventID& eventID() const { return eventID_;}
      LuminosityBlockNumber_t luminosityBlockNumber() const { return eventID_.luminosityBlock();}
      const Timestamp& time() const {return time_; }
      
      bool operator==(const IOVSyncValue& iRHS) const {
	 return comparable(iRHS) && doOp<std::equal_to>(iRHS);
      }
      bool operator!=(const IOVSyncValue& iRHS) const {
	return (!comparable(iRHS)) || doOp<std::not_equal_to>(iRHS);
      }
      
      bool operator<(const IOVSyncValue& iRHS) const {
         return doOp<std::less>(iRHS);
      }
      bool operator<=(const IOVSyncValue& iRHS) const {
         return doOp<std::less_equal>(iRHS);
      }
      bool operator>(const IOVSyncValue& iRHS) const {
         return doOp<std::greater>(iRHS);
      }
      bool operator>=(const IOVSyncValue& iRHS) const {
         return doOp<std::greater_equal>(iRHS);
      }
      
      /** returns true if comparison operations are possible. Comparisons only fail if
	  a time only value is compared to a run/lumi/event only value.
       */
      bool comparable(const IOVSyncValue& iOther) const {
	return (haveID_==iOther.haveID_) || (haveTime_==iOther.haveTime_);
      }

      // ---------- static member functions --------------------
      static const IOVSyncValue& invalidIOVSyncValue();
      static const IOVSyncValue& endOfTime();
      static const IOVSyncValue& beginOfTime();

      // ---------- member functions ---------------------------

   private:
      //IOVSyncValue(const IOVSyncValue&); // stop default

      //const IOVSyncValue& operator=(const IOVSyncValue&); // stop default
      void throwInvalidComparison() const;
      template< template <typename> class Op >
         bool doOp(const IOVSyncValue& iRHS) const {
            bool returnValue = false;
            if(haveID_ && iRHS.haveID_) {
               if(luminosityBlockNumber()==0 || iRHS.luminosityBlockNumber()==0 || luminosityBlockNumber()==iRHS.luminosityBlockNumber()) {
                  Op<EventID> op;
                  returnValue = op(eventID_, iRHS.eventID_);
               } else {
                  if(iRHS.eventID_.run() == eventID_.run()) {
                     Op<LuminosityBlockNumber_t> op;
                     returnValue = op(luminosityBlockNumber(), iRHS.luminosityBlockNumber());
                  } else {
                     Op<RunNumber_t> op;
                     returnValue = op(eventID_.run(), iRHS.eventID_.run());
                  }
               }

            } else if (haveTime_ && iRHS.haveTime_) {
               Op<Timestamp> op;
               returnValue = op(time_, iRHS.time_);
            } else {
               //error
 	       throwInvalidComparison();
            }
            return returnValue;
         }
         
      // ---------- member data --------------------------------
      EventID eventID_;
      Timestamp time_;
      bool haveID_;
      bool haveTime_;
};

}

#endif
