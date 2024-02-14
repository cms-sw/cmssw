#ifndef DataFormats_VertexSoA_interface_ZVertexSoACollection_h
#define DataFormats_VertexSoA_interface_ZVertexSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ZVertexSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ZVertexHost, ZVertexDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<ZVertexDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ZVertexDevice<TDevice> const& deviceData) {
      ZVertexHost hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("ZVertexSoACollection: I'm copying to host.\n");
#endif
      return hostData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ZVertexSoACollection, ZVertexHost);

#endif  // DataFormats_VertexSoA_interface_ZVertexSoACollection_h
