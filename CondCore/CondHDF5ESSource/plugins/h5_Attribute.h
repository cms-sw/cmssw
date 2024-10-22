#ifndef CondCore_CondHDF5ESSource_h5_Attribute_h
#define CondCore_CondHDF5ESSource_h5_Attribute_h
// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     Attribute
//
/**\class Attribute Attribute.h "Attribute.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 14:37:32 GMT
//

// system include files
#include <string>
#include "hdf5.h"

// user include files

// forward declarations

namespace cms::h5 {

  class Attribute {
  public:
    Attribute(hid_t, std::string const&);
    ~Attribute();

    Attribute(const Attribute&) = delete;             // stop default
    Attribute& operator=(const Attribute&) = delete;  // stop default
    Attribute(Attribute&&) = delete;                  // stop default
    Attribute& operator=(Attribute&&) = delete;       // stop default

    // ---------- const member functions ---------------------
    std::string readString() const;
    uint32_t readUInt32() const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------
    hid_t id_;
  };
}  // namespace cms::h5
#endif
