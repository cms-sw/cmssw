#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Hits_test.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testTrackingRecHitSoA {

    template <typename TrackerTraits>
    struct TestFillKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackingRecHitSoAView<TrackerTraits> soa) const {
        const uint32_t i(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t j(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        if (cms::alpakatools::once_per_grid(acc)) {
          soa.offsetBPIX2() = 22;
          soa[10].xLocal() = 1.11;
        }

        soa[i].iphi() = i % 10;
        soa.hitsLayerStart()[j] = j;
      }
    };

    template <typename TrackerTraits>
    struct ShowKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackingRecHitSoAConstView<TrackerTraits> soa) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("nbins = %d\n", soa.phiBinner().nbins());
          printf("offsetBPIX = %d\n", soa.offsetBPIX2());
          printf("nHits = %d\n", soa.metadata().size());
          //printf("hitsModuleStart[28] = %d\n", soa[28].hitsModuleStart());
        }

        // can be increased to soa.nHits() for debugging
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, 10)) {
          printf("iPhi %d -> %d\n", i, soa[i].iphi());
        }
      }
    };

    template <typename TrackerTraits>
    void runKernels(TrackingRecHitSoAView<TrackerTraits>& view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(view.metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel<TrackerTraits>{}, view);
      alpaka::exec<Acc1D>(queue, workDiv, ShowKernel<TrackerTraits>{}, view);
    }

    template void runKernels<pixelTopology::Phase1>(TrackingRecHitSoAView<pixelTopology::Phase1>& view, Queue& queue);
    template void runKernels<pixelTopology::Phase2>(TrackingRecHitSoAView<pixelTopology::Phase2>& view, Queue& queue);

  }  // namespace testTrackingRecHitSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
