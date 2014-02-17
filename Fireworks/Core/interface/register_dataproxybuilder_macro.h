#ifndef Fireworks_Core_register_dataproxybuilder_macro_h
#define Fireworks_Core_register_dataproxybuilder_macro_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     register_dataproxybuilder_macro
//
/**

   Description: Adds needed methods to a ProxyBuilder

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 15:31:20 EDT 2008
// $Id: register_dataproxybuilder_macro.h,v 1.8 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <stdlib.h> 
#include "boost/lexical_cast.hpp"
#include "FWCore/Utilities/interface/TypeWithDict.h"

// forward declarations

#define REGISTER_PROXYBUILDER_METHODS() \
   const std::string& typeName() const { return classTypeName(); } \
   const std::string& view() const { return classView();} \
   const std::string& purpose() const { return classPurpose();} \
   static const std::string& classRegisterTypeName(); \
   static const std::string& classTypeName(); \
   static const std::string& classPurpose(); \
   static const std::string& classView()   \

#define CONCATENATE_HIDDEN(a,b) a ## b
#define CONCATENATE(a,b) CONCATENATE_HIDDEN(a,b)

#define DEFINE_PROXYBUILDER_METHODS(_builder_,_type_,_purpose_,_view_)	\
   const std::string& _builder_::classTypeName() { \
      static std::string s_type = edm::TypeWithDict(typeid(_type_)).name(); \
      return s_type;} \
   const std::string& _builder_::classRegisterTypeName() { \
      static std::string s_type( typeid(_type_).name() ); \
      return s_type;} \
   const std::string& _builder_::classView(){ \
      static  std::string s_view( boost::lexical_cast<std::string>(_view_));    \
      return s_view;}  \
   const std::string& _builder_::classPurpose(){ \
      static std::string s_purpose(_purpose_); return s_purpose;} enum {CONCATENATE(dummy_proxybuilder_methods_, __LINE__)}

#endif

