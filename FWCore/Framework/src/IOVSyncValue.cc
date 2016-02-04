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
IOVSyncValue::IOVSyncValue(): eventID_(), time_(),
haveID_(true), haveTime_(true)
{
}

IOVSyncValue::IOVSyncValue(const EventID& iID) : eventID_(iID), time_(),
haveID_(true), haveTime_(false)
{
}

IOVSyncValue::IOVSyncValue(const Timestamp& iTime) : eventID_(), time_(iTime),
haveID_(false), haveTime_(true)
{
}

IOVSyncValue::IOVSyncValue(const EventID& iID, const Timestamp& iTime) :
eventID_(iID), time_(iTime),
haveID_(true), haveTime_(true)
{
}

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
void 
IOVSyncValue::throwInvalidComparison() const {
  throw cms::Exception("InvalidIOVSyncValueComparison")
    <<"Attempted to compare a time-only and a run/lumi/event-only IOVSyncValue. Please report this error to the framework experts.";
}

//
// static member functions
//
const IOVSyncValue&
IOVSyncValue::invalidIOVSyncValue() {
   static IOVSyncValue s_invalid;
   return s_invalid;
}
const IOVSyncValue&
IOVSyncValue::endOfTime() {
   static IOVSyncValue s_endOfTime(EventID(0xFFFFFFFFUL, LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
                                   Timestamp::endOfTime());
   return s_endOfTime;
}
const IOVSyncValue&
IOVSyncValue::beginOfTime() {
   static IOVSyncValue s_beginOfTime(EventID(1,0,0), Timestamp::beginOfTime());
   return s_beginOfTime;
}
}
