#ifndef CondCore_HDF5ESSource_IOVSyncValue_h
#define CondCore_HDF5ESSource_IOVSyncValue_h
// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     IOVSyncValue
//
/**\class IOVSyncValue IOVSyncValue.h "IOVSyncValue.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 14:38:35 GMT
//

// system include files
#include <vector>

// user include files

// forward declarations

namespace cond::hdf5 {
  struct IOVSyncValue {
    unsigned int high_;
    unsigned int low_;
  };

  inline bool operator<(IOVSyncValue const& iLHS, IOVSyncValue const& iRHS) {
    if (iLHS.high_ < iRHS.high_)
      return true;
    if (iLHS.high_ > iRHS.high_)
      return false;
    return iLHS.low_ < iRHS.low_;
  }

  std::vector<IOVSyncValue>::const_iterator findMatchingFirst(std::vector<IOVSyncValue> const&, IOVSyncValue);
}  // namespace cond::hdf5
#endif
