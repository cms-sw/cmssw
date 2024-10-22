#ifndef CondCore_CondHDF5ESSource_h5_File_h
#define CondCore_CondHDF5ESSource_h5_File_h
// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :      File
//
/**\class  File  File.h " File.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 14:36:39 GMT
//

// system include files
#include <string>
#include <memory>

#include "hdf5.h"

// user include files

// forward declarations

namespace cms::h5 {
  class Group;
  class DataSet;
  class Attribute;

  class File {
  public:
    enum CtrOption { kReadOnly };

    File(std::string const& iName, CtrOption);
    ~File();

    File(const File&) = delete;             // stop default
    File& operator=(const File&) = delete;  // stop default
    File(File&&) = delete;                  // stop default
    File& operator=(File&&) = delete;       // stop default

    // ---------- const member functions ---------------------
    std::shared_ptr<Group> findGroup(std::string const& iName) const;
    std::shared_ptr<DataSet> findDataSet(std::string const& iName) const;
    std::shared_ptr<Attribute> findAttribute(std::string const& iName) const;

    std::shared_ptr<Group> derefGroup(hobj_ref_t iRef) const;
    std::shared_ptr<DataSet> derefDataSet(hobj_ref_t iRef) const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------
    hid_t id_;
  };
}  // namespace cms::h5

#endif
