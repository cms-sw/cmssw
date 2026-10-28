#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "testPhase2StripCPEPosition.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition {

  namespace {
    struct PositionKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    phase2StripCPE::ModuleParams params,
                                    ClusterInput const* clusters,
                                    phase2StripCPE::LocalPosition* positions,
                                    uint32_t size) const {
        for (uint32_t index : cms::alpakatools::uniform_elements(acc, size)) {
          auto const& cluster = clusters[index];
          positions[index] = phase2StripCPE::localPosition(params, cluster.firstStrip, cluster.column, cluster.size);
        }
      }
    };
  }  // namespace

  void localPositions(Queue& queue,
                      phase2StripCPE::ModuleParams const& params,
                      ClusterInput const* clusters,
                      phase2StripCPE::LocalPosition* positions,
                      uint32_t size) {
    auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(size, 256), 256);
    alpaka::exec<Acc1D>(queue, workDiv, PositionKernel{}, params, clusters, positions, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition
