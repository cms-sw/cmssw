#ifndef DataFormats_VertexSoA_interface_ZVertexSoACollection_h
#define DataFormats_VertexSoA_interface_ZVertexSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ZVertexSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ZVertexHost, ZVertexDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ZVertexSoACollection, ZVertexHost);

#endif  // DataFormats_VertexSoA_interface_ZVertexSoACollection_h
