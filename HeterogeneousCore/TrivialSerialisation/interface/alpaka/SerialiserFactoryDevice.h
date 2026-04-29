#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "DataFormats/Common/interface/DeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Serialiser.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {
  using SerialiserFactoryDevice = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

// Helper macro to define Serialiser plugins.
//
// TYPE_DEVICE is the inner product type (e.g. PortableDeviceCollection<...>),
// not wrapped in DeviceProduct, and without ALPAKA_ACCELERATOR_NAMESPACE::
// attached to it (it is attached here).
//
// TYPE_HOST is the type that was passed to DEFINE_TRIVIAL_SERIALISER_PLUGIN
// when registering this type in the non-alpaka TrivialSerialisation factory. It
// is required to match a host type with a device serialiser. The H to D product
// transformation can then be registered through this device serialiser.
//
// The plugin is registered under three keys: 1. EDM_STRINGIZE(TYPE_DEVICE):
// human-readable, more suitable for python configurations; 2. mangled typeid
// name; 3. EDM_STRINGIZE(TYPE_HOST): human-readable host type name. The mapping
// between each TYPE_HOST to its corresponding device serialiser is required
// because the mechanism to register a H->D transform cannot live in the
// non-alpaka Serialiser<TYPE_HOST>.
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(TYPE_HOST, TYPE_DEVICE)                                   \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                                \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>,  \
                    EDM_STRINGIZE(TYPE_DEVICE));                                                               \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     typeid(edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>).name());            \
  DEFINE_EDM_PLUGIN3(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     EDM_STRINGIZE(TYPE_HOST))

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
