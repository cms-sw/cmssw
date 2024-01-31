#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace testTrackingRecHitSoA {

    template <typename TrackerTraits>
    class TestFillKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, TrackingRecHitSoAView<TrackerTraits> soa) const {
        const uint32_t i(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t j(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        if (i == 0 and j == 0) {
          soa.offsetBPIX2() = 22;
          soa[10].xLocal() = 1.11;
        }

        soa[i].iphi() = i % 10;
        soa.hitsLayerStart()[j] = j;
      }
    };

    template <typename TrackerTraits>
    class ShowKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, TrackingRecHitSoAConstView<TrackerTraits> soa) const {
        const uint32_t i(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t j(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        if (i == 0 and j == 0) {
          printf("nbins = %d \n", soa.phiBinner().nbins());
          printf("offsetBPIX %d ->%d \n", i, soa.offsetBPIX2());
          printf("nHits %d ->%d \n", i, soa.metadata().size());
          //printf("hitsModuleStart %d ->%d \n", i, soa.hitsModuleStart().at(28));
        }

        if (i < 10)  // can be increased to soa.nHits() for debugging
          printf("iPhi %d ->%d \n", i, soa[i].iphi());
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
