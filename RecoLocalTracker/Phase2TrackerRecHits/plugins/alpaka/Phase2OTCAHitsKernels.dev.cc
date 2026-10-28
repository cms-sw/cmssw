#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "Phase2OTCAHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  using namespace cms::alpakatools;

  namespace {
    // one thread per hit: the Phase2StripCPE position and errors, and the global position relative to the beam spot
    struct MakeCAHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    Phase2OTCAModuleParamsSoA::ConstView modules,
                                    Cluster const* clusters,
                                    BeamSpotPosition beamSpot,
                                    uint16_t firstDetectorIndex,
                                    ::reco::TrackingRecHitView hits) const {
        for (uint32_t index : uniform_elements(acc, hits.metadata().size())) {
          const Cluster cluster = clusters[index];
          auto const module = modules[cluster.module];
          auto const local =
              phase2StripCPE::localPosition(module.cpe(), cluster.firstStrip, cluster.column, cluster.size);
          float globalX, globalY, globalZ;
          module.frame().toGlobal(local.x, local.y, globalX, globalY, globalZ);
          // in double precision: the same values as from the legacy rechits on the host
          const double hitX = globalX - beamSpot.x;
          const double hitY = globalY - beamSpot.y;
          const double hitZ = globalZ - beamSpot.z;
          auto hit = hits[index];
          hit.xLocal() = local.x;
          hit.yLocal() = local.y;
          hit.xerrLocal() = module.xerrLocal();
          hit.yerrLocal() = module.yerrLocal();
          hit.xGlobal() = hitX;
          hit.yGlobal() = hitY;
          hit.zGlobal() = hitZ;
          hit.rGlobal() = std::sqrt(hitX * hitX + hitY * hitY);
          hit.iphi() = unsafe_atan2s<7>(static_cast<float>(hitY), static_cast<float>(hitX));
          hit.chargeAndStatus() = SiPixelHitStatusAndCharge{};  // charge 0, every status bit 0
          hit.clusterSizeX() = -1;
          hit.clusterSizeY() = -1;
          hit.detectorIndex() = static_cast<uint16_t>(firstDetectorIndex + cluster.module);
        }
      }
    };
  }  // namespace

  void makeCAHits(Queue& queue,
                  Phase2OTCAModuleParamsSoA::ConstView modules,
                  Cluster const* clusters,
                  BeamSpotPosition beamSpot,
                  uint16_t firstDetectorIndex,
                  ::reco::TrackingRecHitView hits) {
    const uint32_t nHits = hits.metadata().size();
    constexpr uint32_t threadsPerBlock = 256;
    auto const workDiv = make_workdiv<Acc1D>(divide_up_by(nHits, threadsPerBlock), threadsPerBlock);
    alpaka::exec<Acc1D>(queue, workDiv, MakeCAHitsKernel{}, modules, clusters, beamSpot, firstDetectorIndex, hits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits
