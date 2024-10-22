#ifndef CondCore_HDF5ESSource_Record_h
#define CondCore_HDF5ESSource_Record_h
// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     Record
//
/**\class Record Record.h "Record.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 14:36:37 GMT
//

// system include files
#include <vector>
#include <string>

// user include files
#include "IOVSyncValue.h"
#include "DataProduct.h"

// forward declarations

namespace cond::hdf5 {
  struct Record {
    std::string name_;
    std::vector<IOVSyncValue> iovFirsts_;
    std::vector<IOVSyncValue> iovLasts_;
    std::vector<DataProduct> dataProducts_;
    bool iovIsRunLumi_ = true;
  };
}  // namespace cond::hdf5

#endif
