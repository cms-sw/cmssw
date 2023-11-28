// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     Group
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 15:26:23 GMT
//

// system include files
#include <cassert>

// user include files
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
  Group::Group(hid_t iParentID, std::string const& iName) : id_(H5Gopen2(iParentID, iName.c_str(), H5P_DEFAULT)) {
    if (id_ < 0) {
      throw cms::Exception("UnknownH5Group") << "unable to find group " << iName;
    }
  }

  Group::Group(hid_t iParentID, const void* iRef) : id_(H5Rdereference2(iParentID, H5P_DEFAULT, H5R_OBJECT, iRef)) {
    if (id_ < 0) {
      throw cms::Exception("BadH5GroupRef") << "unable to dereference Group from parent " << iParentID;
    }
  }

  // Group::Group(const Group& rhs)
  // {
  //    // do actual copying here;
  // }

  Group::~Group() { H5Gclose(id_); }

  //
  // assignment operators
  //
  // const Group& Group::operator=(const Group& rhs)
  // {
  //   //An exception safe implementation is
  //   Group temp(rhs);
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
  std::shared_ptr<Group> Group::findGroup(std::string const& iName) const {
    return std::make_shared<Group>(id_, iName);
  }
  std::shared_ptr<DataSet> Group::findDataSet(std::string const& iName) const {
    return std::make_shared<DataSet>(id_, iName);
  }

  std::shared_ptr<Attribute> Group::findAttribute(std::string const& iName) const {
    return std::make_shared<Attribute>(id_, iName);
  }

  std::shared_ptr<Group> Group::derefGroup(hobj_ref_t iRef) const { return std::make_shared<Group>(id_, &iRef); }

  std::shared_ptr<DataSet> Group::derefDataSet(hobj_ref_t iRef) const { return std::make_shared<DataSet>(id_, &iRef); }

  std::string Group::name() const {
    ssize_t name_size = H5Iget_name(id_, nullptr, 0);

    size_t actual_name_size = static_cast<size_t>(name_size) + 1;
    std::unique_ptr<char[]> buffer(new char[actual_name_size]);
    H5Iget_name(id_, buffer.get(), actual_name_size);

    return std::string(buffer.get());
  }

  std::size_t Group::getNumObjs() const {
    H5G_info_t ginfo;  // Group information

    herr_t ret_value = H5Gget_info(id_, &ginfo);
    assert(ret_value >= 0);
    return (ginfo.nlinks);
  }

  std::string Group::getObjnameByIdx(std::size_t idx) const {
    // call H5Lget_name_by_idx with name as NULL to get its length
    ssize_t name_len = H5Lget_name_by_idx(id_, ".", H5_INDEX_NAME, H5_ITER_INC, idx, nullptr, 0, H5P_DEFAULT);
    assert(name_len >= 0);

    // The actual size is the cast value + 1 for the terminal ASCII NUL
    // (unfortunate in/out type sign mismatch)
    size_t actual_name_len = static_cast<size_t>(name_len) + 1;

    std::unique_ptr<char[]> buffer(new char[actual_name_len]);

    (void)H5Lget_name_by_idx(id_, ".", H5_INDEX_NAME, H5_ITER_INC, idx, buffer.get(), actual_name_len, H5P_DEFAULT);

    return std::string(buffer.get());
  }

  //
  // static member functions
  //
}  // namespace cms::h5
