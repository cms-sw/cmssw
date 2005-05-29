#ifndef EVENTSETUP_TIMESTAMP_H
#define EVENTSETUP_TIMESTAMP_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class:      Timestamp
// 
/**\class Timestamp Timestamp.h Core/CoreFramework/interface/Timestamp.h

 Description: Defines an instance in time

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:23:05 EST 2005
//

// system include files

// user include files

// forward declarations
namespace edm {
class Timestamp
{

   public:
      ///This is a temporary argument until we determine what should be used
      Timestamp(unsigned long iValue) ;
      virtual ~Timestamp();

       unsigned long value() const { return time_; }

      // ---------- const member functions ---------------------
      bool operator==(const Timestamp& iRHS) const {
         return time_ == iRHS.time_;
      }
      bool operator!=(const Timestamp& iRHS) const {
         return time_ != iRHS.time_;
      }

      bool operator<(const Timestamp& iRHS) const {
         return time_ < iRHS.time_;
      }
      bool operator<=(const Timestamp& iRHS) const {
         return time_ <= iRHS.time_;
      }
      bool operator>(const Timestamp& iRHS) const {
         return time_ > iRHS.time_;
      }
      bool operator>=(const Timestamp& iRHS) const {
         return time_ >= iRHS.time_;
      }
      
      // ---------- static member functions --------------------
      static const Timestamp& invalidTimestamp();
      static const Timestamp& endOfTime();
      static const Timestamp& beginOfTime();
      
      // ---------- member functions ---------------------------

   private:
      //Timestamp( const Timestamp& ); // allow default

      //const Timestamp& operator=( const Timestamp& ); // allow default

      // ---------- member data --------------------------------
      unsigned long time_;
      
};

}
#endif /* EVENTSETUP_TIMESTAMP_H */
