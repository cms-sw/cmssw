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
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;
  namespace testTrackingRecHitSoA {

    struct TestFillKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::reco::TrackingBlocksSoAView soa) const {
        const uint32_t i(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t j(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        if (cms::alpakatools::once_per_grid(acc)) {
          soa.trackingHits().offsetBPIX2() = 22;
          soa.trackingHits()[10].xLocal() = 1.11;
        }

        soa.trackingHits()[i].iphi() = i % 10;
        soa.hitModules()[j].moduleStart() = j;
      }
    };

    struct ShowKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::reco::TrackingBlocksSoAConstView soa) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("offsetBPIX = %d\n", soa.trackingHits().offsetBPIX2());
          printf("nHits = %d\n", soa.trackingHits().metadata().size());
          printf("hitsModuleStart[28] = %d\n", soa.hitModules()[28].moduleStart());
        }

        // can be increased to soa.nHits() for debugging
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, soa.trackingHits().metadata().size())) {
          printf("iPhi %d -> %d\n", i, soa.trackingHits()[i].iphi());
          printf("x %d -> %.2f \n", i, soa.trackingHits()[i].xLocal());
        }
      }
    };

    void runKernels(::reco::TrackingBlocksSoAView& view, Queue& queue) {
      uint32_t items = 64;
      uint32_t groups = divide_up_by(view.trackingHits().metadata().size(), items);
      auto workDiv = make_workdiv<Acc1D>(groups, items);
      alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, view);
      alpaka::exec<Acc1D>(queue, workDiv, ShowKernel{}, view);
    }

  }  // namespace testTrackingRecHitSoA
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
