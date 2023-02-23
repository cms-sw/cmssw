#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitProducerKernel.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

namespace {
  // Get subdetector encoded in detId to narrow the range of reference table values to search
  // https://cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
  constexpr uint32_t getSubdet(uint32_t detId) {
    return ((detId >> DetId::kSubdetOffset) & DetId::kSubdetMask);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0163
  constexpr uint32_t getDepth(uint32_t detId) {
    return ((detId >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0148
  constexpr uint32_t getIetaAbs(uint32_t detId) {
    return ((detId >> HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0157
  constexpr uint32_t getIphi(uint32_t detId) {
    return (detId & HcalDetId::kHcalPhiMask2);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
  constexpr int getZside(uint32_t detId) {
    return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1));
  }
}



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

      // TODO get these from config via ESProducer
      const float thresholdE_HB[4] = {0.1, 0.2, 0.3, 0.3};
      const float thresholdE_HE[7] = {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, num_recHits)) {
        const uint32_t detId = recHits[i].detId();
        const uint32_t subdet = getSubdet(detId);
        const uint32_t depth = getDepth(detId);
        const float energy = recHits[i].energy();

        float threshold = 9999.;
        if (subdet == HcalBarrel) {
          threshold = thresholdE_HB[depth - 1];
        } else if (subdet == HcalEndcap) {
          threshold = thresholdE_HE[depth - 1];
        } else {
          printf("Rechit with detId %u has invalid subdetector %u!\n", detId, subdet);
        }

        if (energy >= threshold) {
          const int32_t j = alpaka::atomicAdd(acc, &num_pfRecHits, 1, alpaka::hierarchy::Blocks{});
          pfRecHits[j].detId() = detId;
          pfRecHits[j].energy() = recHits[i].energy();
          pfRecHits[j].time() = recHits[i].time();

          pfRecHits[j].depth() = depth;

          if (subdet == HcalBarrel)
            pfRecHits[j].layer() = PFLayer::HCAL_BARREL1;
          else if (subdet == HcalEndcap)
            pfRecHits[j].layer() = PFLayer::HCAL_ENDCAP;
          else
            pfRecHits[j].layer() = PFLayer::NONE;
          
          //pfRecHits[i].neighbours() = {0, 0, 0, 0, 0, 0, 0, 0};
        }
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