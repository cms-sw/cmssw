#ifndef RecoLocalTracker_SiPixelRecHits_SiStripRecHitSoAKernel_h
#define RecoLocalTracker_SiPixelRecHits_SiStripRecHitSoAKernel_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace hitkernels {
    using namespace cms::alpakatools;

    template <typename TrackerTraits>
    class SiStripRecHitSoAKernel {
      using StripHits = TrackingRecHitsSoACollection<TrackerTraits>;
      using StripHitsHost = TrackingRecHitHost<TrackerTraits>;

    public:
      SiStripRecHitSoAKernel() = default;
      ~SiStripRecHitSoAKernel() = default;

      SiStripRecHitSoAKernel(const SiStripRecHitSoAKernel&) = delete;
      SiStripRecHitSoAKernel(SiStripRecHitSoAKernel&&) = delete;
      SiStripRecHitSoAKernel& operator=(const SiStripRecHitSoAKernel&) = delete;
      SiStripRecHitSoAKernel& operator=(SiStripRecHitSoAKernel&&) = delete;

      StripHits fillHitsAsync(StripHitsHost const& hits_h, Queue queue) const;
    };
  }  // namespace hitkernels
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_SiStripRecHitSoAKernel_h
