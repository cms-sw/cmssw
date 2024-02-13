#ifndef DataFormats_VertexSoA_ZVertexHost_H
#define DataFormats_VertexSoA_ZVertexHost_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

// This alias is needed to call the SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES macro without commas.
using ZVertexHostSoABase = PortableHostCollection2<reco::ZVertexSoA, reco::ZVertexTracksSoA>;

template <int32_t NVTX, int32_t NTRK>
class ZVertexHostSoA : public ZVertexHostSoABase {
public:
  ZVertexHostSoA() = default;

  // Constructor which specifies the queue
  template <typename TQueue>
  explicit ZVertexHostSoA(TQueue queue)
      : PortableHostCollection2<reco::ZVertexSoA, reco::ZVertexTracksSoA>({{NVTX, NTRK}}, queue) {}

  // Constructor which specifies the DevHost
  explicit ZVertexHostSoA(alpaka_common::DevHost const& host)
      : PortableHostCollection2<reco::ZVertexSoA, reco::ZVertexTracksSoA>({{NVTX, NTRK}}, host) {}
};

using ZVertexHost = ZVertexHostSoA<zVertex::MAXVTX, zVertex::MAXTRACKS>;

#endif  // DataFormats_VertexSoA_ZVertexHost_H
