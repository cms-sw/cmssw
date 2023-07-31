// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     Attribute
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 15:26:38 GMT
//

// system include files

// user include files
#include "h5_Attribute.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//

namespace cms::h5 {
  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  Attribute::Attribute(hid_t iParentID, std::string const& iName)
      : id_(H5Aopen(iParentID, iName.c_str(), H5P_DEFAULT)) {
    if (id_ < 1) {
      throw cms::Exception("UnknownH5Attribute") << "unknown attribute " << iName;
    }
  }

  // Attribute::Attribute(const Attribute& rhs)
  // {
  //    // do actual copying here;
  // }

  Attribute::~Attribute() { H5Aclose(id_); }

  //
  // assignment operators
  //
  // const Attribute& Attribute::operator=(const Attribute& rhs)
  // {
  //   //An exception safe implementation is
  //   Attribute temp(rhs);
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
  std::string Attribute::readString() const {
    // Prepare and call C API to read attribute.
    char* strg_C;

    hid_t attr_type = H5Tcopy(H5T_C_S1);
    (void)H5Tset_size(attr_type, H5T_VARIABLE);

    // Read attribute, no allocation for variable-len string; C library will
    herr_t ret_value = H5Aread(id_, attr_type, &strg_C);
    H5Tclose(attr_type);

    if (ret_value < 0) {
      throw cms::Exception("H5AttributeReadStrinFailed") << " failed to read string from attribute";
    }

    // Get string from the C char* and release resource allocated by C API
    std::string strg = strg_C;
    free(strg_C);

    return strg;
  }

  //
  // static member functions
  //
}  // namespace cms::h5
