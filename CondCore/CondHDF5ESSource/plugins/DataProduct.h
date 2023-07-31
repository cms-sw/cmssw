#ifndef CondCore_HDF5ESSource_DataProduct_h
#define CondCore_HDF5ESSource_DataProduct_h
// -*- C++ -*-
//
// Package:     CondCore/HDF5ESSource
// Class  :     DataProduct
//
/**\class DataProduct DataProduct.h "DataProduct.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 20 Jun 2023 14:36:44 GMT
//

// system include files
#include <string>
#include <vector>
#include <H5Cpp.h>

// user include files

// forward declarations

namespace cond::hdf5 {
  struct DataProduct {
    DataProduct(std::string iName, std::string iType) : name_(std::move(iName)), type_(std::move(iType)) {}
    std::string name_;
    std::string type_;
    std::vector<hobj_ref_t> payloadForIOVs_;
  };
}  // namespace cond::hdf5
#endif
