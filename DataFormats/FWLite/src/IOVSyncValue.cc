// -*- C++ -*-
//
// Package:     FWLite
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
#include "DataFormats/FWLite/interface/IOVSyncValue.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

//
// constants, enums and typedefs
//
namespace fwlite {

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  IOVSyncValue::IOVSyncValue() : eventID_(), time_(), haveID_(true), haveTime_(true) {}

  IOVSyncValue::IOVSyncValue(const edm::EventID& iID) : eventID_(iID), time_(), haveID_(true), haveTime_(false) {}

  IOVSyncValue::IOVSyncValue(const edm::Timestamp& iTime) : eventID_(), time_(iTime), haveID_(false), haveTime_(true) {}

  IOVSyncValue::IOVSyncValue(const edm::EventID& iID, const edm::Timestamp& iTime)
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

  //
  // static member functions
  //
  const IOVSyncValue& IOVSyncValue::invalidIOVSyncValue() {
    static const IOVSyncValue s_invalid;
    return s_invalid;
  }
  const IOVSyncValue& IOVSyncValue::endOfTime() {
    static const IOVSyncValue s_endOfTime(
        edm::EventID(0xFFFFFFFFUL, edm::LuminosityBlockID::maxLuminosityBlockNumber(), edm::EventID::maxEventNumber()),
        edm::Timestamp::endOfTime());
    return s_endOfTime;
  }
  const IOVSyncValue& IOVSyncValue::beginOfTime() {
    static const IOVSyncValue s_beginOfTime(edm::EventID(1, 0, 0), edm::Timestamp::beginOfTime());
    return s_beginOfTime;
  }
}  // namespace fwlite
