#ifndef HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h
#define HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Serialiser.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"

namespace ngt {
  // Plugin factory for host-product Serialisers, keyed by typeid name.
  using SerialiserFactory = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ngt

// Register a Serialiser plugin for a host product type.
//
// The plugin is registered under both the mangled typeid name and
// EDM_STRINGIZE(TYPE). EDM_STRINGIZE(TYPE) is more human-readable, and thus
// more suitable for Python configuration files.
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN(TYPE)                                           \
  DEFINE_EDM_PLUGIN(ngt::SerialiserFactory, ngt::Serialiser<TYPE>, typeid(TYPE).name()); \
  DEFINE_EDM_PLUGIN2(ngt::SerialiserFactory, ngt::Serialiser<TYPE>, EDM_STRINGIZE(TYPE))

#endif  // HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h
