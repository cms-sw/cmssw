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
// $Id: register_itemaccessorbase_macro.h,v 1.3 2012/06/26 22:13:03 wmtan Exp $
//

// system include files
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files

// forward declarations
#define REGISTER_FWITEMACCESSOR_METHODS() \
   const std::string& typeName() const { return classTypeName(); } \
   const std::string& purpose() const { return classPurpose();} \
   static const std::string& classRegisterTypeName(); \
   static const std::string& classTypeName(); \
   static const std::string& classPurpose()

#define CONCATENATE_HIDDEN(a,b) a ## b
#define CONCATENATE(a,b) CONCATENATE_HIDDEN(a,b)

#define DEFINE_FWITEMACCESSOR_METHODS(_accessor_,_type_,_purpose_) \
   const std::string& _accessor_::classTypeName() { \
      static std::string s_type = edm::TypeWithDict(typeid(_type_)).name(); \
      return s_type;} \
   const std::string& _accessor_::classRegisterTypeName() { \
      static std::string s_type(typeid(_type_).name()); \
      return s_type;} \
   const std::string& _accessor_::classPurpose() { \
      static std::string s_purpose(_purpose_); return s_purpose;} enum {CONCATENATE(dummy_itemaccessor_methods_, __LINE__)}

#define DEFINE_TEMPLATE_FWITEMACCESSOR_METHODS(_accessor_,_type_,_purpose_) \
   template<> const std::string& _accessor_::classTypeName() { \
      static std::string s_type = edm::TypeWithDict(typeid(_type_)).name(); \
      return s_type;} \
   template<> const std::string& _accessor_::classRegisterTypeName() { \
      static std::string s_type(typeid(_type_).name()); \
      return s_type;} \
   template<> const std::string& _accessor_::classPurpose() { \
      static std::string s_purpose(_purpose_); return s_purpose;} enum {CONCATENATE(dummy_itemaccessor_methods_, __LINE__)}

#endif
