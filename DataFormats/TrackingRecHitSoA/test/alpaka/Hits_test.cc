#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include <alpaka/alpaka.hpp>
#include <unistd.h>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testTrackingRecHitSoA {

    template <typename TrackerTraits>
    void runKernels(TrackingRecHitSoAView<TrackerTraits>& hits, Queue& queue);

  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // inner scope to deallocate memory before destroying the queue
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[pixelTopology::Phase1::numberOfModules + 1];

    for (size_t i = 0; i < pixelTopology::Phase1::numberOfModules + 1; i++) {
      moduleStart[i] = i * 2;
    }
    TrackingRecHitsSoACollection<pixelTopology::Phase1> tkhit(nHits, offset, &moduleStart[0], queue);

    testTrackingRecHitSoA::runKernels<pixelTopology::Phase1>(tkhit.view(), queue);
    alpaka::wait(queue);
  }
  return 0;
}
