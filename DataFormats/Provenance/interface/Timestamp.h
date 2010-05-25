#ifndef DataFormats_Provenance_Timestamp_h
#define DataFormats_Provenance_Timestamp_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class:      Timestamp
//
/**\class Timestamp Timestamp.h DataFormats/Provenance/interface/Timestamp.h

 Description: Defines an instance in time from the Online system

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
   typedef unsigned long long TimeValue_t;

class Timestamp {

   public:
      explicit Timestamp(TimeValue_t iValue);
      Timestamp();

      virtual ~Timestamp();

      /// Time in seconds since January 1, 1970.
      unsigned int
      unixTime() const {
        return timeHigh_;
      }

      /// Nanoseconds offset within second
      unsigned int
      nanosecondOffset() const {
        return timeLow_;
      }

      TimeValue_t value() const;

      // ---------- const member functions ---------------------
      bool operator==(Timestamp const& iRHS) const {
         return timeHigh_ == iRHS.timeHigh_ &&
         timeLow_ == iRHS.timeLow_;
      }
      bool operator!=(Timestamp const& iRHS) const {
         return !(*this == iRHS);
      }

      bool operator<(Timestamp const& iRHS) const {
         if(timeHigh_ == iRHS.timeHigh_) {
            return timeLow_ < iRHS.timeLow_;
         }
         return timeHigh_ < iRHS.timeHigh_;
      }
      bool operator<=(Timestamp const& iRHS) const {
         if(timeHigh_ == iRHS.timeHigh_) {
            return timeLow_ <= iRHS.timeLow_;
         }
         return timeHigh_ <= iRHS.timeHigh_;
      }
      bool operator>(Timestamp const& iRHS) const {
         if(timeHigh_ == iRHS.timeHigh_) {
            return timeLow_ > iRHS.timeLow_;
         }
         return timeHigh_ > iRHS.timeHigh_;
      }
      bool operator>=(Timestamp const& iRHS) const {
         if(timeHigh_ == iRHS.timeHigh_) {
            return timeLow_ >= iRHS.timeLow_;
         }
         return timeHigh_ >= iRHS.timeHigh_;
      }

      // ---------- static member functions --------------------
      static Timestamp const& invalidTimestamp();
      static Timestamp const& endOfTime();
      static Timestamp const& beginOfTime();

      // ---------- member functions ---------------------------

   private:
      //Timestamp(Timestamp const&); // allow default

      //Timestamp const& operator=(Timestamp const&); // allow default

      // ---------- member data --------------------------------
      // ROOT does not support ULL
      //TimeValue_t time_;
      unsigned int timeLow_;
      unsigned int timeHigh_;
};

}
#endif
