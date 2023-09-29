#ifndef CondFormats_SerializationHelper_SerializationHelperFactory_h
#define CondFormats_SerializationHelper_SerializationHelperFactory_h
// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     SerializationHelperFactory
//
/**\class SerializationHelperFactory SerializationHelperFactory.h "CondFormats/SerializationHelper/interface/SerializationHelperFactory.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 14:55:17 GMT
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondFormats/SerializationHelper/interface/SerializationHelper.h"

// forward declarations

namespace cond::serialization {
  using SerializationHelperFactory = edmplugin::PluginFactory<SerializationHelperBase*()>;
}

#define DEFINE_COND_CLASSNAME(type_)                    \
  namespace cond::serialization {                       \
    template <>                                         \
    struct ClassName<type_> {                           \
      constexpr static std::string_view kName = #type_; \
    };                                                  \
  }

#define DEFINE_COND_SERIAL_REGISTER_PLUGIN(type_) \
  DEFINE_COND_CLASSNAME(type_)                    \
  DEFINE_EDM_PLUGIN(                              \
      cond::serialization::SerializationHelperFactory, cond::serialization::SerializationHelper<type_>, #type_)

#define DEFINE_COND_SERIAL_REGISTER_PLUGIN_INIT(type_, init_)                                                   \
  namespace cond::serialization {                                                                               \
    template <>                                                                                                 \
    struct ClassName<type_> {                                                                                   \
      constexpr static std::string_view kName = #type_;                                                         \
    };                                                                                                          \
  }                                                                                                             \
  using EDM_PLUGIN_SYM(SerializationHelper, __LINE__) = cond::serialization::SerializationHelper<type_, init_>; \
  DEFINE_EDM_PLUGIN(                                                                                            \
      cond::serialization::SerializationHelperFactory, EDM_PLUGIN_SYM(SerializationHelper, __LINE__), #type_)

#endif
