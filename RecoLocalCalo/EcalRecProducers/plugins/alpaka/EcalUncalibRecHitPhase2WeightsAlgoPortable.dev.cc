#include <alpaka/alpaka.hpp>

#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "EcalUncalibRecHitPhase2WeightsAlgoPortable.h"
#include "EcalUncalibRecHitPhase2WeightsStruct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::weights {
  using namespace cms::alpakatools;

  class Phase2WeightsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  EcalUncalibRecHitPhase2Weights const* weightsObj,
                                  EcalDigiPhase2DeviceCollection::ConstView digisDev,
                                  EcalUncalibratedRecHitDeviceCollection::View uncalibratedRecHitsDev) const {
      constexpr int nsamples = ecalPh2::sampleSize;
      auto const nchannels = digisDev.size();
      // one thread sets the output collection size scalar
      if (once_per_grid(acc)) {
        uncalibratedRecHitsDev.size() = digisDev.size();
      }

      auto const* weightsdata = weightsObj->weights.data();
      auto const* timeWeightsdata = weightsObj->timeWeights.data();
      //divide the grid into uniform elements
      for (auto tx : uniform_elements(acc, nchannels)) {
        bool g1 = false;
        // Avoid false-positive Wdangling-reference
        const auto& digisDevTx = digisDev[tx];
        const auto& digi = digisDevTx.data();
        auto recHit = uncalibratedRecHitsDev[tx];
        recHit.amplitude() = 0;
        recHit.jitter() = 0;
        for (int s = 0; s < nsamples; ++s) {
          const auto sample = digi[s];
          const auto trace =
              (static_cast<float>(ecalLiteDTU::adc(sample))) * ecalPh2::gains[ecalLiteDTU::gainId(sample)];
          recHit.amplitude() += (trace * weightsdata[s]);
          recHit.jitter() += (trace * timeWeightsdata[s]);
          if (ecalLiteDTU::gainId(sample) == 1)
            g1 = true;
          recHit.outOfTimeAmplitudes()[s] = 0.;
        }
        recHit.amplitudeError() = 1.0f;
        recHit.id() = digisDev.id()[tx];
        recHit.flags() = 0;
        recHit.pedestal() = 0.;
        recHit.jitterError() = 0.;
        recHit.chi2() = 0.;
        recHit.aux() = 0;
        if (g1) {
          recHit.flags() = 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;
        }
      }  //if within nchannels
    }  //kernel}
  };

  void phase2Weights(EcalDigiPhase2DeviceCollection const& digis,
                     EcalUncalibratedRecHitDeviceCollection& uncalibratedRecHits,
                     EcalUncalibRecHitPhase2Weights const* weightsObj,
                     Queue& queue) {
    // use 64 items per group (arbitrary value, a reasonable starting point)
    uint32_t items = 64;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(digis->metadata().size(), items);
    //create the work division
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    //launch the kernel
    alpaka::exec<Acc1D>(
        queue, workDiv, Phase2WeightsKernel{}, weightsObj, digis.const_view(), uncalibratedRecHits.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::weights
