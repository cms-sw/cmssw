#ifndef CondCore_CondHDF5ESSource_DataSet_h
#define CondCore_CondHDF5ESSource_DataSet_h
// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     DataSet
//
/**\class DataSet DataSet.h "DataSet.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 14:37:25 GMT
//

// system include files
#include <memory>
#include <string>
#include <vector>

#include "hdf5.h"

// user include files
#include "IOVSyncValue.h"

// forward declarations

namespace cms::h5 {
  class File;
  class Group;
  class Attribute;

  class DataSet {
  public:
    DataSet(hid_t iParentID, std::string const& iName);
    DataSet(hid_t iParentID, const void* iRef);
    ~DataSet();

    DataSet(const DataSet&) = delete;                   // stop default
    const DataSet& operator=(const DataSet&) = delete;  // stop default

    // ---------- const member functions ---------------------
    std::shared_ptr<Attribute> findAttribute(std::string const& iName) const;

    std::vector<hobj_ref_t> readRefs() const;
    std::vector<char> readBytes() const;
    std::vector<cond::hdf5::IOVSyncValue> readSyncValues() const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    std::size_t size() const;

    // ---------- member data --------------------------------
    hid_t id_;
  };

}  // namespace cms::h5
#endif
