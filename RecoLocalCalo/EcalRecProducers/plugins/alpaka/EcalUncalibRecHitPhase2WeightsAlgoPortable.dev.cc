#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EcalUncalibRecHitPhase2WeightsAlgoPortable.h"

#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace ecal {
    namespace weights {

      using namespace cms::alpakatools;

      class Phase2WeightsKernel {
      public:
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                      double const *weightsdata,
                                      double const *timeWeightsdata,
                                      EcalDigiPhase2DeviceCollection::ConstView digisDev,
                                      EcalUncalibratedRecHitDeviceCollection::View uncalibratedRecHitsDev) const;
      };

      template <typename TAcc>
      ALPAKA_FN_ACC void Phase2WeightsKernel::operator()(
          TAcc const &acc,
          double const *weightsData,
          double const *timeWeightsdata,
          EcalDigiPhase2DeviceCollection::ConstView digisDev,
          EcalUncalibratedRecHitDeviceCollection::View uncalibratedRecHitsDev) const {
        constexpr int nsamples = EcalDataFrame_Ph2::MAXSAMPLES;
        auto const nchannels = digisDev.size();
        // one thread sets the output collection size scalar
        if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] == 0) {
          uncalibratedRecHitsDev.size() = digisDev.size();
        }

        auto *amplitude = uncalibratedRecHitsDev.amplitude();
        auto *jitter = uncalibratedRecHitsDev.jitter();
        const auto *digis = digisDev.data();
        //calculate the first and the stride
        const auto first = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                           alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                               alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        const auto stride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] *
                            alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];

        for (auto tx = first; tx < nchannels; tx += stride) {
          bool g1 = false;
          auto const did = DetId{digisDev.id()[tx]};
          amplitude[tx] = 0;
          jitter[tx] = 0;
          for (int sample = 0; sample < nsamples; ++sample) {
            const auto digi = digis[tx][sample];
            const auto trace = (static_cast<float>(ecalLiteDTU::adc(digi))) * ecalPh2::gains[ecalLiteDTU::gainId(digi)];
            amplitude[tx] += (trace * *(weightsData + sample));
            jitter[tx] += (trace * *(timeWeightsdata + sample));
            if (ecalLiteDTU::gainId(digi) == 1)
              g1 = true;
            uncalibratedRecHitsDev.outOfTimeAmplitudes()[tx][sample] = 0.;
          }
          uncalibratedRecHitsDev.amplitudeError()[tx] = 1.0f;
          uncalibratedRecHitsDev.id()[tx] = did.rawId();
          uncalibratedRecHitsDev.flags()[tx] = 0;
          uncalibratedRecHitsDev.pedestal()[tx] = 0.;
          uncalibratedRecHitsDev.jitterError()[tx] = 0.;
          uncalibratedRecHitsDev.chi2()[tx] = 0.;
          uncalibratedRecHitsDev.aux()[tx] = 0;
          if (g1) {
            uncalibratedRecHitsDev.flags()[tx] = 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;
          }
        }  //if within nchannels
      }  //kernel}

      void phase2Weights(EcalDigiPhase2DeviceCollection const &digis,
                         EcalUncalibratedRecHitDeviceCollection &uncalibratedRecHits,
                         const cms::alpakatools::host_buffer<double[]> &weights,
                         const cms::alpakatools::host_buffer<double[]> &timeWeights,
                         Queue &queue) {
        //create device buffers for the weights and copy the data from host to the device
        auto weightsDev = make_device_buffer<double[]>(queue, ecalPh2::sampleSize);
        auto timeWeightsDev = make_device_buffer<double[]>(queue, ecalPh2::sampleSize);
        alpaka::memcpy(queue, weightsDev, weights);
        alpaka::memcpy(queue, timeWeightsDev, timeWeights);

        // use 64 items per group (arbitrary value, a reasonable starting point)
        uint32_t items = 64;
        // use as many groups as needed to cover the whole problem
        uint32_t groups = divide_up_by(digis->metadata().size(), items);
        //create the work division
        auto workDiv = make_workdiv<Acc1D>(groups, items);
        //launch the kernel
        alpaka::exec<Acc1D>(queue,
                            workDiv,
                            Phase2WeightsKernel{},
                            weightsDev.data(),
                            timeWeightsDev.data(),
                            digis.const_view(),
                            uncalibratedRecHits.view());
      }

    }  // namespace weights
  }  // namespace ecal
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
