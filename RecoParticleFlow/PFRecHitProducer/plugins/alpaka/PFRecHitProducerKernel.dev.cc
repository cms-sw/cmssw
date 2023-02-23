#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitProducerKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class PFRecHitProducerKernelImpl {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const CaloRecHitDeviceCollection::ConstView recHits, int32_t num_recHits,
                                  PFRecHitDeviceCollection::View pfRecHits) const {
      // global index of the thread within the grid
      const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

      // set this only once in the whole kernel grid
      int& num_pfRecHits = alpaka::declareSharedVar<int,__COUNTER__>(acc);
      if (thread == 0) {
        num_pfRecHits = 0;
      }
      alpaka::syncBlockThreads(acc);

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, num_recHits)) {

        // TODO real filtering
        if(i % 2 == 1)
          continue;

        const int32_t j = alpaka::atomicAdd(acc, &num_pfRecHits, 1, alpaka::hierarchy::Blocks{});
        pfRecHits[j].detId() = recHits[i].detId();
        pfRecHits[j].energy() = recHits[i].energy();
        pfRecHits[j].time() = recHits[i].time();
        pfRecHits[j].depth() = 0;
        //pfRecHits[i].neighbours() = {0, 0, 0, 0, 0, 0, 0, 0};
      }

      alpaka::syncBlockThreads(acc);

      if (thread == 0) {
        pfRecHits.size() = num_pfRecHits;
      }
    }
  };

  void PFRecHitProducerKernel::execute(Queue& queue, const CaloRecHitDeviceCollection& recHits, PFRecHitDeviceCollection& pfRecHits) const {
    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    const uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    const uint32_t groups = 1;//divide_up_by(recHits->metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, PFRecHitProducerKernelImpl{}, recHits.view(), recHits->metadata().size(), pfRecHits.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE