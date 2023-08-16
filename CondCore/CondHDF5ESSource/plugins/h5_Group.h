#ifndef CondCore_CondHDF5ESSource_h5_Group_h
#define CondCore_CondHDF5ESSource_h5_Group_h
// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     Group
//
/**\class Group Group.h "Group.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 14:37:18 GMT
//

// system include files
#include <memory>
#include <string>

#include "hdf5.h"

// user include files

// forward declarations

namespace cms::h5 {
  class File;
  class DataSet;
  class Attribute;

  class Group {
  public:
    explicit Group(hid_t, std::string const&);
    explicit Group(hid_t, const void* iRef);

    ~Group();

    Group(const Group&) = delete;             // stop default
    Group& operator=(const Group&) = delete;  // stop default
    Group(Group&&) = delete;                  // stop default
    Group& operator=(Group&&) = delete;       // stop default

    // ---------- const member functions ---------------------
    std::shared_ptr<Group> findGroup(std::string const& iName) const;
    std::shared_ptr<DataSet> findDataSet(std::string const& iName) const;
    std::shared_ptr<Attribute> findAttribute(std::string const& iName) const;

    std::shared_ptr<Group> derefGroup(hobj_ref_t iRef) const;
    std::shared_ptr<DataSet> derefDataSet(hobj_ref_t iRef) const;

    std::string name() const;

    std::size_t getNumObjs() const;
    std::string getObjnameByIdx(std::size_t) const;
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------

    hid_t id_;
  };
}  // namespace cms::h5
#endif
