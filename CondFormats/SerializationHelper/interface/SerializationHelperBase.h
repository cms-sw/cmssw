#ifndef CondFormats_SerializationHelper_SerializationHelperBase_h
#define CondFormats_SerializationHelper_SerializationHelperBase_h
// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     SerializationHelperBase
//
/**\class SerializationHelperBase SerializationHelperBase.h "CondFormats/SerializationHelper/interface/SerializationHelperBase.h"

 Description: abstract base class for helper that can serialize/deserialize conditions data products in a type hidden way.

 Usage:
    This is used in conjunction with the SerializationHelperFactory.

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 14:45:56 GMT
//

// system include files
#include <string_view>

// user include files
#include "CondFormats/SerializationHelper/interface/unique_void_ptr.h"

// forward declarations

namespace cond::serialization {
  class SerializationHelperBase {
  public:
    SerializationHelperBase() = default;
    virtual ~SerializationHelperBase() = default;

    SerializationHelperBase(const SerializationHelperBase&) = delete;                   // stop default
    const SerializationHelperBase& operator=(const SerializationHelperBase&) = delete;  // stop default

    // ---------- const member functions ---------------------
    virtual unique_void_ptr deserialize(std::streambuf&, const std::string_view iClassName) const = 0;

    //returns name of type serialized. This is needed for polymorphism storage
    virtual std::string_view serialize(std::streambuf&, void const*) const = 0;
    virtual const std::type_info& type() const = 0;
  };
}  // namespace cond::serialization
#endif
