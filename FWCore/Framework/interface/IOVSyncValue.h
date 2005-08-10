#ifndef FRAMEWORK_IOVSYNCVALUE_H
#define FRAMEWORK_IOVSYNCVALUE_H
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
// $Id: IOVSyncValue.h,v 1.1 2005/08/04 14:25:47 chrjones Exp $
//

// system include files
#include <functional>

// user include files
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"

// forward declarations

namespace edm {
class IOVSyncValue
{

   public:
      IOVSyncValue();
      //virtual ~IOVSyncValue();
      explicit IOVSyncValue(const EventID& iID);
      explicit IOVSyncValue( const Timestamp& iTime);
      IOVSyncValue( const EventID& iID, const Timestamp& iID);

      // ---------- const member functions ---------------------
      const EventID& eventID() const { return eventID_;}
      const Timestamp& time() const {return time_; }
      
      bool operator==(const IOVSyncValue& iRHS) const {
         return doOp<std::equal_to>( iRHS);
      }
      bool operator!=(const IOVSyncValue& iRHS) const {
         return doOp<std::not_equal_to>( iRHS);
      }
      
      bool operator<(const IOVSyncValue& iRHS) const {
         return doOp<std::less>( iRHS);
      }
      bool operator<=(const IOVSyncValue& iRHS) const {
         return doOp<std::less_equal>( iRHS);
      }
      bool operator>(const IOVSyncValue& iRHS) const {
         return doOp<std::greater>( iRHS);
      }
      bool operator>=(const IOVSyncValue& iRHS) const {
         return doOp<std::greater_equal>( iRHS);
      }
      
      // ---------- static member functions --------------------
      static const IOVSyncValue& invalidIOVSyncValue();
      static const IOVSyncValue& endOfTime();
      static const IOVSyncValue& beginOfTime();

      // ---------- member functions ---------------------------

   private:
      //IOVSyncValue( const IOVSyncValue& ); // stop default

      //const IOVSyncValue& operator=( const IOVSyncValue& ); // stop default
      template< template <typename> class Op >
         bool doOp(const IOVSyncValue& iRHS ) const {
            bool returnValue = false;
            if( haveID_ && iRHS.haveID_ ) {
               Op<EventID> op;
               returnValue = op(eventID_, iRHS.eventID_);
            } else if ( haveTime_ && iRHS.haveTime_ ) {
               Op<Timestamp> op;
               returnValue = op(time_, iRHS.time_);
            } else {
               //error
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

#endif /* FRAMEWORK_IOVSYNCVALUE_H */
