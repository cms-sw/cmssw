#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Serialiser.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {
  using SerialiserFactoryDevice = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

// Helper macro to define Serialiser plugins.
//
// TYPE_DEVICE is the device type alias (e.g. sistrip::SiStripClusterDevice),
// without ALPAKA_ACCELERATOR_NAMESPACE:: attached to it (it is attached here).
//
// TYPE_HOST is the type that was passed to DEFINE_TRIVIAL_SERIALISER_PLUGIN
// when registering this type in the non-alpaka TrivialSerialisation factory. It
// is required to match a host type with a device serialiser. The H to D product
// transformation can then be registered through this device serialiser.
//
// The plugin is registered under two keys:
//
//   1. mangled typeid name of TYPE_HOST: used to look up the device serialiser
//      for a host type.
//   2. EDM_STRINGIZE(TYPE_DEVICE): used to look up the device serialiser
#define DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(TYPE_HOST, TYPE_DEVICE)                                      \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                                \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>,  \
                    typeid(TYPE_HOST).name());                                                                 \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     EDM_STRINGIZE(TYPE_DEVICE))
#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
