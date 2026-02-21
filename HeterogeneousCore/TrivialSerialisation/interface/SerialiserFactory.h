#ifndef HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h
#define HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h

#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Serialiser.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ngt {
  // Plugin factory for host-product Serialisers, keyed by typeid name.
  using SerialiserFactory = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ngt

// Register a Serialiser plugin for a host product type.
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN(TYPE) \
  DEFINE_EDM_PLUGIN(ngt::SerialiserFactory, ngt::Serialiser<TYPE>, typeid(TYPE).name())

#endif  // HeterogeneousCore_TrivialSerialisation_interface_SerialiserFactory_h
