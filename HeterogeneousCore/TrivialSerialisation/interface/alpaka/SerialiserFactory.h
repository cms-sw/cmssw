#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactory_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactory_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Serialiser.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {
  using SerialiserFactoryPortable = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

// Helper macro to define Serialiser plugins with a human-readable name string.
//
// TYPE is the inner product type (e.g. PortableDeviceCollection<...>), not
// wrapped in DeviceProduct, and without ALPAKA_ACCELERATOR_NAMESPACE:: attached
// to it (it is attached here). The plugin is registered under both the mangled
// typeid name and a human-readable NAME. NAME is mainly intended for Python
// configuration files.
#define DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(TYPE, NAME)                                           \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryPortable,                       \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE>,  \
                    typeid(edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::TYPE>).name());             \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryPortable,                      \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE>, \
                     NAME)

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactory_h
