#ifndef RecoTracker_LSTCore_interface_alpaka_LSTInputCollection_h
#define RecoTracker_LSTCore_interface_alpaka_LSTInputCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/LSTInputDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using LSTInputCollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ::lst::LSTInputHostCollection, ::lst::LSTInputDeviceCollection<Device>>;
}  // namespace lst

#endif