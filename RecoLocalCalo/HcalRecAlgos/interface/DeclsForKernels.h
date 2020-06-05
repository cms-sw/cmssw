#ifndef RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
#define RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"

#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalLUTCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedEffectivePedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedEffectivePedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimeCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIETypesGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMParametersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMCharacteristicsGPU.h"

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include <optional>
#include <functional>

namespace hcal {
  namespace mahi {

    struct ConditionsProducts {
      HcalGainWidthsGPU::Product const& gainWidths;
      HcalGainsGPU::Product const& gains;
      HcalLUTCorrsGPU::Product const& lutCorrs;
      HcalConvertedPedestalWidthsGPU::Product const& pedestalWidths;
      HcalConvertedEffectivePedestalWidthsGPU::Product const& effectivePedestalWidths;
      HcalConvertedPedestalsGPU::Product const& pedestals;
      HcalQIECodersGPU::Product const& qieCoders;
      HcalRecoParamsWithPulseShapesGPU::Product const& recoParams;
      HcalRespCorrsGPU::Product const& respCorrs;
      HcalTimeCorrsGPU::Product const& timeCorrs;
      HcalQIETypesGPU::Product const& qieTypes;
      HcalSiPMParametersGPU::Product const& sipmParameters;
      HcalSiPMCharacteristicsGPU::Product const& sipmCharacteristics;
      HcalConvertedPedestalsGPU::Product const* convertedEffectivePedestals;
      HcalTopology const* topology;
      HcalDDDRecConstants const* recConstants;
      uint32_t offsetForHashes;
    };

    struct ConfigParameters {
      uint32_t maxChannels;
      uint32_t maxTimeSamples;
      uint32_t kprep1dChannelsPerBlock;
      int sipmQTSShift;
      int sipmQNTStoSum;
      int firstSampleShift;
      bool useEffectivePedestals;

      float meanTime;
      float timeSigmaSiPM, timeSigmaHPD;
      float ts4Thresh;

      std::vector<int> pulseOffsets;
      int* pulseOffsetsDevice = nullptr;

      std::array<uint32_t, 3> kernelMinimizeThreads;

      // FIXME:
      //   - add "getters" to HcalTimeSlew calib formats
      //   - add ES Producer to consume what is produced above not to replicate.
      //   which ones to use is hardcoded, therefore no need to send those to the device
      bool applyTimeSlew;
      float tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew;
    };

    struct OutputDataGPU {
      RecHitCollection<common::ViewStoragePolicy> recHits;

      void allocate(ConfigParameters const& config) {
        cudaCheck(cudaMalloc((void**)&recHits.energy, config.maxChannels * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recHits.chi2, config.maxChannels * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recHits.energyM0, config.maxChannels * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recHits.timeM0, config.maxChannels * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recHits.did, config.maxChannels * sizeof(uint32_t)));
      }

      void deallocate(ConfigParameters const& config) {
        cudaCheck(cudaFree(recHits.energy));
        cudaCheck(cudaFree(recHits.chi2));
        cudaCheck(cudaFree(recHits.energyM0));
        cudaCheck(cudaFree(recHits.timeM0));
        cudaCheck(cudaFree(recHits.did));
      }
    };

    struct ScratchDataGPU {
      float *amplitudes = nullptr, *noiseTerms = nullptr;
      float *pulseMatrices = nullptr, *pulseMatricesM = nullptr, *pulseMatricesP = nullptr;
      int8_t* soiSamples = nullptr;

      // TODO: properly allocate for NSAMPLES VS NPULSES
      void allocate(ConfigParameters const& config) {
        cudaCheck(cudaMalloc((void**)&amplitudes, sizeof(float) * config.maxChannels * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&noiseTerms, sizeof(float) * config.maxChannels * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatrices,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatricesM,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatricesP,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&soiSamples, sizeof(int8_t) * config.maxChannels));
      }

      void deallocate(ConfigParameters const& config) {
        if (amplitudes) {
          cudaCheck(cudaFree(amplitudes));
          cudaCheck(cudaFree(noiseTerms));
          cudaCheck(cudaFree(pulseMatrices));
          cudaCheck(cudaFree(pulseMatricesM));
          cudaCheck(cudaFree(pulseMatricesP));
          cudaCheck(cudaFree(soiSamples));
        }
      }
    };

    struct InputDataGPU {
      DigiCollection<Flavor01, common::ViewStoragePolicy> const& f01HEDigis;
      DigiCollection<Flavor5, common::ViewStoragePolicy> const& f5HBDigis;
      DigiCollection<Flavor3, common::ViewStoragePolicy> const& f3HBDigis;
    };

  }  // namespace mahi
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
