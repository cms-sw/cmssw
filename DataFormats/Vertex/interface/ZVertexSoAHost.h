#ifndef DataFormats_Vertex_ZVertexSoAHost_H
#define DataFormats_Vertex_ZVertexSoAHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <int32_t S>
class ZVertexSoAHost : public PortableHostCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  ZVertexSoAHost() = default;

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit ZVertexSoAHost(TQueue queue) : PortableHostCollection<ZVertexSoAHeterogeneousLayout<>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit ZVertexSoAHost(alpaka_common::DevHost const& host)
      : PortableHostCollection<ZVertexSoAHeterogeneousLayout<>>(S, host) {}
};

using namespace ::zVertex;
using ZVertexHost = ZVertexSoAHost<MAXTRACKS>;

#endif  // DataFormats_Vertex_ZVertexSoAHost_H
