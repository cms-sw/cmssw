// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     DataSet
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 15:26:30 GMT
//

// system include files
#include <cassert>

// user include files
#include "h5_DataSet.h"
#include "h5_Attribute.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace {
  struct Type {
    explicit Type(hid_t iID) : id_(iID) { assert(id_ >= 0); }
    ~Type() noexcept { H5Tclose(id_); }

    void insertMember(const char* iName, size_t offset, hid_t member_id) { H5Tinsert(id_, iName, offset, member_id); }
    hid_t id_;
  };

  struct DataSpace {
    explicit DataSpace(hid_t iID) : id_(iID) { assert(id_ >= 0); }
    ~DataSpace() noexcept { H5Sclose(id_); }
    hid_t id_;
  };
}  // namespace

namespace cms::h5 {
  //
  // constructors and destructor
  //
  DataSet::DataSet(hid_t iParentID, std::string const& iName) : id_(H5Dopen2(iParentID, iName.c_str(), H5P_DEFAULT)) {
    if (id_ < 0) {
      throw cms::Exception("UnknownH5DataSet") << "unable to find dataset " << iName;
    }
  }

  DataSet::DataSet(hid_t iParentID, const void* iRef) : id_(H5Rdereference2(iParentID, H5P_DEFAULT, H5R_OBJECT, iRef)) {
    if (id_ < 0) {
      throw cms::Exception("BadH5DataSetRef") << "unable to derenfence dataset from parent " << iParentID;
    }
  }

  // DataSet::DataSet(const DataSet& rhs)
  // {
  //    // do actual copying here;
  // }

  DataSet::~DataSet() { H5Dclose(id_); }

  //
  // assignment operators
  //
  // const DataSet& DataSet::operator=(const DataSet& rhs)
  // {
  //   //An exception safe implementation is
  //   DataSet temp(rhs);
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
  std::shared_ptr<Attribute> DataSet::findAttribute(std::string const& iName) const {
    return std::make_shared<Attribute>(id_, iName);
  }

  std::size_t DataSet::size() const {
    DataSpace space_id{H5Dget_space(id_)};

    hssize_t num_elements = H5Sget_simple_extent_npoints(space_id.id_);
    assert(num_elements >= 0);

    return num_elements;
  }

  std::size_t DataSet::storageSize() const { return H5Dget_storage_size(id_); }
  std::size_t DataSet::memorySize() const { return size(); }
  uint64_t DataSet::fileOffset() const { return H5Dget_offset(id_); }

  uint32_t DataSet::layout() const {
    auto pl = H5Dget_create_plist(id_);
    auto ret = H5Pget_layout(pl);
    H5Pclose(pl);
    return ret;
  }

  std::vector<hobj_ref_t> DataSet::readRefs() const {
    Type type_id{H5Dget_type(id_)};
    auto class_type = H5Tget_class(type_id.id_);
    if (class_type != H5T_REFERENCE) {
      throw cms::Exception("BadDataSetType") << "asked to read dataset as a ref, but it is a " << class_type;
    }

    std::vector<hobj_ref_t> refs;
    refs.resize(size());

    auto ret_value = H5Dread(id_, H5T_STD_REF_OBJ, H5S_ALL, H5S_ALL, H5P_DEFAULT, refs.data());
    if (ret_value < 0) {
      throw cms::Exception("BadH5Read") << "unable to read ref dataset " << id_;
    }
    return refs;
  }

  std::vector<char> DataSet::readBytes() const {
    Type type_id{H5Dget_type(id_)};
    auto class_type = H5Tget_class(type_id.id_);
    if (class_type != H5T_INTEGER) {
      throw cms::Exception("BadDataSetType") << "asked to read dataset as a byte, but it is a " << class_type;
    }

    std::vector<char> bytes;
    bytes.resize(size());

    auto ret_value = H5Dread(id_, H5T_STD_I8LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data());
    if (ret_value < 0) {
      throw cms::Exception("BadH5Read") << "unable to read bytes dataset " << id_;
    }
    return bytes;
  }

  std::vector<cond::hdf5::IOVSyncValue> DataSet::readSyncValues() const {
    Type sv_type_id{H5Tcreate(H5T_COMPOUND, sizeof(cond::hdf5::IOVSyncValue))};
    sv_type_id.insertMember("high", HOFFSET(cond::hdf5::IOVSyncValue, high_), H5T_NATIVE_UINT32);
    sv_type_id.insertMember("low", HOFFSET(cond::hdf5::IOVSyncValue, low_), H5T_NATIVE_UINT32);

    {
      const Type type_id{H5Dget_type(id_)};
      if (not H5Tequal(sv_type_id.id_, type_id.id_)) {
        throw cms::Exception("BadDataSetType")
            << "asked to read dataset as a IOVSyncValue, but it is a " << type_id.id_;
      }
    }

    std::vector<cond::hdf5::IOVSyncValue> syncValues;
    syncValues.resize(size());

    auto ret_value = H5Dread(id_, sv_type_id.id_, H5S_ALL, H5S_ALL, H5P_DEFAULT, syncValues.data());
    if (ret_value < 0) {
      throw cms::Exception("BadH5Read") << "unable to read IOVSyncValue dataset " << id_;
    }
    return syncValues;
  }

  //
  // static member functions
  //
}  // namespace cms::h5
