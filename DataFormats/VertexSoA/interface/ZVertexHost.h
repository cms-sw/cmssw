#ifndef DataFormats_VertexSoA_ZVertexHost_H
#define DataFormats_VertexSoA_ZVertexHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <int32_t S>
class ZVertexHostSoA : public PortableHostCollection<reco::ZVertexSoA> {
public:
  ZVertexHostSoA() = default;

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit ZVertexHostSoA(TQueue queue) : PortableHostCollection<reco::ZVertexSoA>(S, queue) {}

  // Constructor which specifies the DevHost
  explicit ZVertexHostSoA(alpaka_common::DevHost const& host) : PortableHostCollection<reco::ZVertexSoA>(S, host) {}
};

//using namespace ::zVertex;
using ZVertexHost = ZVertexHostSoA<zVertex::MAXTRACKS>;

#endif  // DataFormats_VertexSoA_ZVertexHost_H
