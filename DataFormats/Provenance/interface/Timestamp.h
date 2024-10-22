#ifndef DataFormats_Provenance_Timestamp_h
#define DataFormats_Provenance_Timestamp_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class:      Timestamp
//
/**\class Timestamp Timestamp.h DataFormats/Provenance/interface/Timestamp.h

 Description: Defines an instance in time from the Online system

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:23:05 EST 2005
//

#include <limits>

namespace edm {
  typedef unsigned long long TimeValue_t;

  class Timestamp {
    static const TimeValue_t kLowMask = 0xFFFFFFFF;

  public:
    explicit Timestamp(TimeValue_t iValue)
        : timeLow_(static_cast<unsigned int>(kLowMask & iValue)), timeHigh_(static_cast<unsigned int>(iValue >> 32)) {}

    Timestamp() : timeLow_(invalidTimestamp().timeLow_), timeHigh_(invalidTimestamp().timeHigh_) {}

    /// Time in seconds since January 1, 1970.
    unsigned int unixTime() const { return timeHigh_; }

    /// Microseconds offset within second
    unsigned int microsecondOffset() const { return timeLow_; }

    TimeValue_t value() const {
      TimeValue_t returnValue = timeHigh_;
      returnValue = returnValue << 32;
      returnValue += timeLow_;
      return returnValue;
    }

    // ---------- const member functions ---------------------
    bool operator==(Timestamp const& iRHS) const { return timeHigh_ == iRHS.timeHigh_ && timeLow_ == iRHS.timeLow_; }
    bool operator!=(Timestamp const& iRHS) const { return !(*this == iRHS); }

    bool operator<(Timestamp const& iRHS) const {
      if (timeHigh_ == iRHS.timeHigh_) {
        return timeLow_ < iRHS.timeLow_;
      }
      return timeHigh_ < iRHS.timeHigh_;
    }
    bool operator<=(Timestamp const& iRHS) const {
      if (timeHigh_ == iRHS.timeHigh_) {
        return timeLow_ <= iRHS.timeLow_;
      }
      return timeHigh_ <= iRHS.timeHigh_;
    }
    bool operator>(Timestamp const& iRHS) const {
      if (timeHigh_ == iRHS.timeHigh_) {
        return timeLow_ > iRHS.timeLow_;
      }
      return timeHigh_ > iRHS.timeHigh_;
    }
    bool operator>=(Timestamp const& iRHS) const {
      if (timeHigh_ == iRHS.timeHigh_) {
        return timeLow_ >= iRHS.timeLow_;
      }
      return timeHigh_ >= iRHS.timeHigh_;
    }

    // ---------- static member functions --------------------
    static Timestamp invalidTimestamp() { return Timestamp(0); }
    static Timestamp endOfTime() { return Timestamp(std::numeric_limits<TimeValue_t>::max()); }
    static Timestamp beginOfTime() { return Timestamp(1); }

  private:
    // ---------- member data --------------------------------
    // ROOT does not support ULL
    //TimeValue_t time_;
    unsigned int timeLow_;
    unsigned int timeHigh_;
  };

}  // namespace edm
#endif
