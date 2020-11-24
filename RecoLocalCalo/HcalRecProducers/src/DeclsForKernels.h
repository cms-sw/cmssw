#ifndef RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h
#define RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h

#include <functional>
#include <optional>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CondFormats/DataRecord/interface/HcalCombinedRecordsGPU.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalGainsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIECodersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypesGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParametersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrsGPU.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalMahiPulseOffsetsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"

namespace hcal {
  namespace reconstruction {

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
      HcalMahiPulseOffsetsGPU::Product const& pulseOffsets;
      std::vector<int, cms::cuda::HostAllocator<int>> const& pulseOffsetsHost;
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

      std::array<uint32_t, 3> kernelMinimizeThreads;

      // FIXME:
      //   - add "getters" to HcalTimeSlew calib formats
      //   - add ES Producer to consume what is produced above not to replicate.
      //   which ones to use is hardcoded, therefore no need to send those to the device
      bool applyTimeSlew;
      float tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew;
    };

    struct OutputDataGPU {
      RecHitCollection<::calo::common::DevStoragePolicy> recHits;

      void allocate(ConfigParameters const& config, cudaStream_t cudaStream) {
        recHits.energy = cms::cuda::make_device_unique<float[]>(config.maxChannels, cudaStream);
        recHits.chi2 = cms::cuda::make_device_unique<float[]>(config.maxChannels, cudaStream);
        recHits.energyM0 = cms::cuda::make_device_unique<float[]>(config.maxChannels, cudaStream);
        recHits.timeM0 = cms::cuda::make_device_unique<float[]>(config.maxChannels, cudaStream);
        recHits.did = cms::cuda::make_device_unique<uint32_t[]>(config.maxChannels, cudaStream);
      }
    };

    struct ScratchDataGPU {
      cms::cuda::device::unique_ptr<float[]> amplitudes, noiseTerms, pulseMatrices, pulseMatricesM, pulseMatricesP;
      cms::cuda::device::unique_ptr<int8_t[]> soiSamples;
    };

    struct InputDataGPU {
      DigiCollection<Flavor1, ::calo::common::DevStoragePolicy> const& f01HEDigis;
      DigiCollection<Flavor5, ::calo::common::DevStoragePolicy> const& f5HBDigis;
      DigiCollection<Flavor3, ::calo::common::DevStoragePolicy> const& f3HBDigis;
    };

  }  // namespace reconstruction
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h
