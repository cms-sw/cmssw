// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     IOVSyncValue
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 18:20:47 GMT
//

// system include files

// user include files
#include "IOVSyncValue.h"

namespace cond::hdf5 {
  std::vector<IOVSyncValue>::const_iterator findMatchingFirst(std::vector<IOVSyncValue> const& iIOVs,
                                                              IOVSyncValue iMatch) {
    auto itFound = std::lower_bound(iIOVs.begin(), iIOVs.end(), iMatch);
    if (itFound == iIOVs.end() or iMatch < *itFound) {
      //need to back up one space
      if (itFound == iIOVs.begin()) {
        return iIOVs.end();
      }
      itFound -= 1;
    }
    return itFound;
  }
}  // namespace cond::hdf5
