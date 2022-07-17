#ifndef Fireworks_Core_register_itemaccessorbase_macro_h
#define Fireworks_Core_register_itemaccessorbase_macro_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     register_itemaccessorbase_macros_h
//
/**

   Description: Adds needed methods to a FWItemAccessorBase 

   Usage:
    <usage>

 */
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 11:31:20 EDT 2010
//

// system include files
#include <string>

// user include files
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/concatenate.h"

// forward declarations

#define REGISTER_FWITEMACCESSOR_METHODS()                         \
  const std::string& typeName() const { return classTypeName(); } \
  const std::string& purpose() const { return classPurpose(); }   \
  static const std::string& classRegisterTypeName();              \
  static const std::string& classTypeName();                      \
  static const std::string& classPurpose()

#define DEFINE_FWITEMACCESSOR_METHODS(_accessor_, _type_, _purpose_)      \
  const std::string& _accessor_::classTypeName() {                        \
    static std::string s_type = edm::TypeWithDict(typeid(_type_)).name(); \
    return s_type;                                                        \
  }                                                                       \
  const std::string& _accessor_::classRegisterTypeName() {                \
    static std::string s_type(typeid(_type_).name());                     \
    return s_type;                                                        \
  }                                                                       \
  const std::string& _accessor_::classPurpose() {                         \
    static std::string s_purpose(_purpose_);                              \
    return s_purpose;                                                     \
  }                                                                       \
  enum { EDM_CONCATENATE(dummy_itemaccessor_methods_, __LINE__) }

#define DEFINE_TEMPLATE_FWITEMACCESSOR_METHODS(_accessor_, _type_, _purpose_) \
  template <>                                                                 \
  const std::string& _accessor_::classTypeName() {                            \
    static std::string s_type = edm::TypeWithDict(typeid(_type_)).name();     \
    return s_type;                                                            \
  }                                                                           \
  template <>                                                                 \
  const std::string& _accessor_::classRegisterTypeName() {                    \
    static std::string s_type(typeid(_type_).name());                         \
    return s_type;                                                            \
  }                                                                           \
  template <>                                                                 \
  const std::string& _accessor_::classPurpose() {                             \
    static std::string s_purpose(_purpose_);                                  \
    return s_purpose;                                                         \
  }                                                                           \
  enum { EDM_CONCATENATE(dummy_itemaccessor_methods_, __LINE__) }

#endif
