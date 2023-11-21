#ifndef DataFormats_Vertex_ZVertexHost_H
#define DataFormats_Vertex_ZVertexHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <int32_t S>
class ZVertexHostSoA : public PortableHostCollection<ZVertexLayout<>> {
public:
  ZVertexHostSoA() = default;

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit ZVertexHostSoA(TQueue queue) : PortableHostCollection<ZVertexLayout<>>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit ZVertexHostSoA(alpaka_common::DevHost const& host) : PortableHostCollection<ZVertexLayout<>>(S, host) {}
};

using namespace ::zVertex;
using ZVertexHost = ZVertexHostSoA<MAXTRACKS>;

#endif  // DataFormats_Vertex_ZVertexHost_H
