// -*- C++ -*-
//
// Package:     Framework
// Class  :     IOVSyncValue
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  3 18:35:35 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

//
// constants, enums and typedefs
//
namespace edm {

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  IOVSyncValue::IOVSyncValue() : eventID_(), time_(), haveID_(true), haveTime_(true) {}

  IOVSyncValue::IOVSyncValue(const EventID& iID) : eventID_(iID), time_(), haveID_(true), haveTime_(false) {}

  IOVSyncValue::IOVSyncValue(const Timestamp& iTime) : eventID_(), time_(iTime), haveID_(false), haveTime_(true) {}

  IOVSyncValue::IOVSyncValue(const EventID& iID, const Timestamp& iTime)
      : eventID_(iID), time_(iTime), haveID_(true), haveTime_(true) {}

  // IOVSyncValue::IOVSyncValue(const IOVSyncValue& rhs)
  // {
  //    // do actual copying here;
  // }

  //IOVSyncValue::~IOVSyncValue()
  //{
  //}

  //
  // assignment operators
  //
  // const IOVSyncValue& IOVSyncValue::operator=(const IOVSyncValue& rhs)
  // {
  //   //An exception safe implementation is
  //   IOVSyncValue temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  //
  // const member functions
  //
  void IOVSyncValue::throwInvalidComparison() const {
    throw cms::Exception("InvalidIOVSyncValueComparison")
        << "Attempted to compare a time-only and a run/lumi/event-only IOVSyncValue. Please report this error to the "
           "framework experts.";
  }

  //
  // static member functions
  //
  const IOVSyncValue& IOVSyncValue::invalidIOVSyncValue() {
    static const IOVSyncValue s_invalid;
    return s_invalid;
  }
  const IOVSyncValue& IOVSyncValue::endOfTime() {
    static const IOVSyncValue s_endOfTime(
        EventID(0xFFFFFFFFUL, LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
        Timestamp::endOfTime());
    return s_endOfTime;
  }
  const IOVSyncValue& IOVSyncValue::beginOfTime() {
    static const IOVSyncValue s_beginOfTime(EventID(1, 0, 0), Timestamp::beginOfTime());
    return s_beginOfTime;
  }

  std::ostream& operator<<(std::ostream& oStream, IOVSyncValue const& iIOV) {
    if (iIOV.haveID_ && iIOV.haveTime_) {
      oStream << "IOVSyncValue{ EventID{" << iIOV.eventID_.run() << ", " << iIOV.eventID_.luminosityBlock() << ", "
              << iIOV.eventID_.event() << "}, Timestamp{" << iIOV.time_.unixTime() << "} }";
    } else if (iIOV.haveID_) {
      oStream << "IOVSyncValue{ EventID{" << iIOV.eventID_.run() << ", " << iIOV.eventID_.luminosityBlock() << ", "
              << iIOV.eventID_.event() << "} }";
    } else if (iIOV.haveTime_) {
      oStream << "IOVSyncValue{ Timestamp{" << iIOV.time_.unixTime() << "} }";
    }
    return oStream;
  }
}  // namespace edm
