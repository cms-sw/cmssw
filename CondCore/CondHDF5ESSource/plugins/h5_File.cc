// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     File
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 15:26:16 GMT
//

// system include files

// user include files
#include "h5_File.h"
#include "h5_Group.h"
#include "h5_DataSet.h"
#include "h5_Attribute.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace cms::h5 {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  File::File(std::string const& iName, CtrOption) : id_(H5Fopen(iName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) {
    if (id_ < 0) {
      throw cms::Exception("FailedHDF5FileOpen") << "failed to open HDF5 file " << iName;
    }
  }

  // File::File(const File& rhs)
  // {
  //    // do actual copying here;
  // }

  File::~File() { H5Fclose(id_); }

  //
  // assignment operators
  //
  // const File& File::operator=(const File& rhs)
  // {
  //   //An exception safe implementation is
  //   File temp(rhs);
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
  std::shared_ptr<Group> File::findGroup(std::string const& iName) const { return std::make_shared<Group>(id_, iName); }
  std::shared_ptr<DataSet> File::findDataSet(std::string const& iName) const {
    return std::make_shared<DataSet>(id_, iName);
  }

  std::shared_ptr<Attribute> File::findAttribute(std::string const& iName) const {
    return std::make_shared<Attribute>(id_, iName);
  }

  std::shared_ptr<Group> File::derefGroup(hobj_ref_t iRef) const { return std::make_shared<Group>(id_, &iRef); }

  std::shared_ptr<DataSet> File::derefDataSet(hobj_ref_t iRef) const { return std::make_shared<DataSet>(id_, &iRef); }

  //
  // static member functions
  //
}  // namespace cms::h5
