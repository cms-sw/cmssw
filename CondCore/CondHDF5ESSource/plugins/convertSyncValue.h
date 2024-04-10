#ifndef CondCore_HDF5ESSource_convertSyncValue_h
#define CondCore_HDF5ESSource_convertSyncValue_h
// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     convertSyncValue
//
/**\class convertSyncValue convertSyncValue.h "convertSyncValue.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 18:26:32 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "IOVSyncValue.h"

// forward declarations

namespace cond::hdf5 {
  inline IOVSyncValue convertSyncValue(edm::IOVSyncValue const& iFrom, bool iIsRunLumi) {
    if (iIsRunLumi) {
      return IOVSyncValue{iFrom.eventID().run(), iFrom.eventID().luminosityBlock()};
    }
    return IOVSyncValue{iFrom.time().unixTime(), iFrom.time().microsecondOffset()};
  }

  inline edm::IOVSyncValue convertSyncValue(IOVSyncValue const& iFrom, bool iIsRunLumi) {
    if (iIsRunLumi) {
      return edm::IOVSyncValue{edm::EventID{iFrom.high_, iFrom.low_, 0}};
    }
    return edm::IOVSyncValue{edm::Timestamp{(static_cast<uint64_t>(iFrom.high_) << 32) + iFrom.low_}};
  }
}  // namespace cond::hdf5

#endif
