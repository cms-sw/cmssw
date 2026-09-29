#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h

#include <string>

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
// TYPE_DEVICE is the device type (e.g. sistrip::SiStripClusterDevice), without
// ALPAKA_ACCELERATOR_NAMESPACE:: attached to it (it is attached here).
//
// TYPE_HOST is the corresponding host type, required by the device serialiser
// to register H to D product transformations.
//
// Depending on the backend, the plugin is registered under the following
// keys:
//
//   1. Mangled typeid of TYPE_HOST: Both on CPU and GPU backends. Used to look
//   up the device serialiser for a host type.
//   2. Mangled typeid of ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE: Only on GPU
//   backends. Used to look up the device serialiser for a device type. This key
//   is omitted on CPU backends, where ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE
//   resolves to TYPE_HOST (which was already registed in 1.)
//   3. EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) + "::" +
//   EDM_STRINGIZE(TYPE_DEVICE): Only on CPU backends. See CMSSW issue #51427.
//
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED || defined ALPAKA_ACC_GPU_HIP_ENABLED
#define DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(TYPE_HOST, TYPE_DEVICE)                                      \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                                \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>,  \
                    typeid(TYPE_HOST).name());                                                                 \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     typeid(ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE).name());
#else
#define DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(TYPE_HOST, TYPE_DEVICE)                                      \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                                \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>,  \
                    typeid(TYPE_HOST).name());                                                                 \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     std::string(EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)) + "::" + EDM_STRINGIZE(TYPE_DEVICE));
#endif

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
