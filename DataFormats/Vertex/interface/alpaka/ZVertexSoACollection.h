#ifndef DataFormats_Vertex_interface_ZVertexSoACollection_h
#define DataFormats_Vertex_interface_ZVertexSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexUtilities.h"
#include "DataFormats/Vertex/interface/ZVertexSoAHost.h"
#include "DataFormats/Vertex/interface/ZVertexSoADevice.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using ZVertexCollection = ZVertexHost;
#else
  using ZVertexCollection = ZVertexDevice<Device>;
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::ZVertexCollection> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::ZVertexCollection const& deviceData) {
      ZVertexHost hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Vertex_interface_ZVertexSoACollection_h
