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
// $Id$
//

// system include files
#include <functional>

// user include files
#include "FWCore/EDProduct/interface/CollisionID.h"

// forward declarations

namespace edm {
class IOVSyncValue
{

   public:
      IOVSyncValue();
      //virtual ~IOVSyncValue();
      explicit IOVSyncValue(const CollisionID& iID) : collisionID_(iID) {} 
      //explicit IOVSyncValue( const Timestamp& iID)
      //IOVSyncValue( const CollisionID& iID, const Timestamp& iID)

      // ---------- const member functions ---------------------
      const CollisionID& collisionID() const { return collisionID_;}
      //const Timestamp& time() const {return time_; }
      
      bool operator==(const IOVSyncValue& iRHS) const {
         return doOp<std::equal_to>( iRHS);
         //return collisionID_ == iRHS.collisionID_;
      }
      bool operator!=(const IOVSyncValue& iRHS) const {
         return doOp<std::not_equal_to>( iRHS);
         //return collisionID_ != iRHS.collisionID_;
      }
      
      bool operator<(const IOVSyncValue& iRHS) const {
         return doOp<std::less>( iRHS);
         //return collisionID_ < iRHS.collisionID_;
      }
      bool operator<=(const IOVSyncValue& iRHS) const {
         return doOp<std::less_equal>( iRHS);
         //return collisionID_ <= iRHS.collisionID_;
      }
      bool operator>(const IOVSyncValue& iRHS) const {
         return doOp<std::greater>( iRHS);
         //return collisionID_ > iRHS.collisionID_;
      }
      bool operator>=(const IOVSyncValue& iRHS) const {
         return doOp<std::greater_equal>( iRHS);
         //return collisionID_ >= iRHS.collisionID_;
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
            //if( haveID_ && iRHS.haveID_ ) {
               Op<CollisionID> op;
               returnValue = op(collisionID_, iRHS.collisionID_);
            //} else if ( haveTime_ && iRHS.haveTime_ ) {
            //   Op<Timestamp> op;
            //   returnValue = op(time_, iRHS.time_);
            //} else {
               //error
            //}
            return returnValue;
         }
         
      // ---------- member data --------------------------------
      CollisionID collisionID_;
      //Timestamp time_;
      //bool haveID_;
      //bool haveTime_;
};

}

#endif /* FRAMEWORK_IOVSYNCVALUE_H */
