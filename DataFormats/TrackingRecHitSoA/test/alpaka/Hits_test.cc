#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testTrackingRecHitSoA {

    template <typename TrackerTraits>
    void runKernels(TrackingRecHitSoAView<TrackerTraits>& hits, Queue& queue);

  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // inner scope to deallocate memory before destroying the queue
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[pixelTopology::Phase1::numberOfModules + 1];

    for (size_t i = 0; i < pixelTopology::Phase1::numberOfModules + 1; ++i) {
      moduleStart[i] = i * 2;
    }
    TrackingRecHitsSoACollection<pixelTopology::Phase1> tkhit(queue, nHits, offset, moduleStart);

    testTrackingRecHitSoA::runKernels<pixelTopology::Phase1>(tkhit.view(), queue);
    tkhit.updateFromDevice(queue);

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED or defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    // requires c++23 to make cms::alpakatools::CopyToHost compile using if constexpr
    // see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
    TrackingRecHitHost<pixelTopology::Phase1> const& host_collection = tkhit;
#else
    TrackingRecHitHost<pixelTopology::Phase1> host_collection =
        cms::alpakatools::CopyToHost<TrackingRecHitDevice<pixelTopology::Phase1, Device> >::copyAsync(queue, tkhit);
#endif
    // wait for the kernel and the potential copy to complete
    alpaka::wait(queue);
    assert(tkhit.nHits() == nHits);
    assert(tkhit.offsetBPIX2() == 22);  // set in the kernel
    assert(tkhit.nHits() == host_collection.nHits());
    assert(tkhit.offsetBPIX2() == host_collection.offsetBPIX2());
  }

  return 0;
}
