#ifndef DataFormats_FWLite_IOVSyncValue_h
#define DataFormats_FWLite_IOVSyncValue_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     IOVSyncValue
// 
/**\class IOVSyncValue IOVSyncValue.h DataFormats/Framework/interface/IOVSyncValue.h

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

namespace fwlite {
class IOVSyncValue
{

   public:
      IOVSyncValue();
      //virtual ~IOVSyncValue();
      explicit IOVSyncValue(const edm::EventID& iID);
      explicit IOVSyncValue(const edm::Timestamp& iTime);
      IOVSyncValue(const edm::EventID& iID, const edm::Timestamp& iTime);

      // ---------- const member functions ---------------------
      const edm::EventID& eventID() const { return eventID_;}
      edm::LuminosityBlockNumber_t luminosityBlockNumber() const { return eventID_.luminosityBlock();}
      const edm::Timestamp& time() const {return time_; }
      
      bool operator==(const IOVSyncValue& iRHS) const {
         return doOp<std::equal_to>(iRHS);
      }
      bool operator!=(const IOVSyncValue& iRHS) const {
         return doOp<std::not_equal_to>(iRHS);
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
      
      // ---------- static member functions --------------------
      static const IOVSyncValue& invalidIOVSyncValue();
      static const IOVSyncValue& endOfTime();
      static const IOVSyncValue& beginOfTime();

      // ---------- member functions ---------------------------

   private:
      //IOVSyncValue(const IOVSyncValue&); // stop default

      //const IOVSyncValue& operator=(const IOVSyncValue&); // stop default
      template< template <typename> class Op >
         bool doOp(const IOVSyncValue& iRHS) const {
            bool returnValue = false;
            if(haveID_ && iRHS.haveID_) {
               if(luminosityBlockNumber()==0 || iRHS.luminosityBlockNumber()==0 || luminosityBlockNumber()==iRHS.luminosityBlockNumber()) {
                  Op<edm::EventID> op;
                  returnValue = op(eventID_, iRHS.eventID_);
               } else {
                  if(iRHS.eventID_.run() == eventID_.run()) {
                     Op<edm::LuminosityBlockNumber_t> op;
                     returnValue = op(luminosityBlockNumber(), iRHS.luminosityBlockNumber());
                  } else {
                     Op<edm::RunNumber_t> op;
                     returnValue = op(eventID_.run(), iRHS.eventID_.run());
                  }
               }

            } else if (haveTime_ && iRHS.haveTime_) {
               Op<edm::Timestamp> op;
               returnValue = op(time_, iRHS.time_);
            } else {
               //error
            }
            return returnValue;
         }
         
      // ---------- member data --------------------------------
      edm::EventID eventID_;
      edm::Timestamp time_;
      bool haveID_;
      bool haveTime_;
};

}

#endif
