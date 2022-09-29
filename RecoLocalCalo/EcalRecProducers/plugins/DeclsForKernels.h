#ifndef RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphasGPU.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitParametersGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPedestalsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariancesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitADCToGeVConstantGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelationGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

#include "EigenMatrixTypes_gpu.h"

struct EcalPulseShape;
class EcalSampleMask;
class EcalTimeBiasCorrections;
struct EcalPulseCovariance;
class EcalDigiCollection;
class EcalXtalGroupId;
class EcalSamplesCorrelation;
class EBDigiCollection;
class EEDigiCollection;

namespace ecal {
  namespace multifit {

    enum class TimeComputationState : char { NotFinished = 0, Finished = 1 };
    enum class MinimizationState : char {
      NotFinished = 0,
      Finished = 1,
      Precomputed = 2,
    };

    //
    struct EventInputDataGPU {
      ecal::DigisCollection<calo::common::DevStoragePolicy> const& ebDigis;
      ecal::DigisCollection<calo::common::DevStoragePolicy> const& eeDigis;
    };

    // parameters have a fixed type
    // Can we go by with single precision
    struct ConfigurationParameters {
      using type = double;
      // device ptrs
      const type *amplitudeFitParametersEB = nullptr, *amplitudeFitParametersEE = nullptr;

      uint32_t timeFitParametersSizeEB, timeFitParametersSizeEE;
      // device ptrs
      const type *timeFitParametersEB = nullptr, *timeFitParametersEE = nullptr;

      type timeFitLimitsFirstEB, timeFitLimitsFirstEE;
      type timeFitLimitsSecondEB, timeFitLimitsSecondEE;

      type timeConstantTermEB, timeConstantTermEE;

      type timeNconstEB, timeNconstEE;

      type amplitudeThreshEE, amplitudeThreshEB;

      type outOfTimeThreshG12pEB, outOfTimeThreshG12mEB;
      type outOfTimeThreshG12pEE, outOfTimeThreshG12mEE;
      type outOfTimeThreshG61pEE, outOfTimeThreshG61mEE;
      type outOfTimeThreshG61pEB, outOfTimeThreshG61mEB;

      std::array<uint32_t, 3> kernelMinimizeThreads;

      bool shouldRunTimingComputation;

      uint32_t maxNumberHitsEB;
      uint32_t maxNumberHitsEE;
    };

    struct EventOutputDataGPU {
      UncalibratedRecHit<::calo::common::DevStoragePolicy> recHitsEB, recHitsEE;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        memoryPool::Deleter deleter =
            memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(cudaStream, memoryPool::onDevice));
        assert(deleter.pool());
        auto const sizeEB = configParameters.maxNumberHitsEB;
        recHitsEB.amplitudesAll =
            memoryPool::cuda::makeBuffer<reco::ComputationScalarType>(sizeEB * EcalDataFrame::MAXSAMPLES, deleter);
        recHitsEB.amplitude = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEB, deleter);
        recHitsEB.chi2 = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEB, deleter);
        recHitsEB.pedestal = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEB, deleter);

        if (configParameters.shouldRunTimingComputation) {
          recHitsEB.jitter = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEB, deleter);
          recHitsEB.jitterError = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEB, deleter);
        }

        recHitsEB.did = memoryPool::cuda::makeBuffer<uint32_t>(sizeEB, deleter);
        recHitsEB.flags = memoryPool::cuda::makeBuffer<uint32_t>(sizeEB, deleter);

        auto const sizeEE = configParameters.maxNumberHitsEE;
        recHitsEE.amplitudesAll =
            memoryPool::cuda::makeBuffer<reco::ComputationScalarType>(sizeEE * EcalDataFrame::MAXSAMPLES, deleter);
        recHitsEE.amplitude = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEE, deleter);
        recHitsEE.chi2 = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEE, deleter);
        recHitsEE.pedestal = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEE, deleter);

        if (configParameters.shouldRunTimingComputation) {
          recHitsEE.jitter = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEE, deleter);
          recHitsEE.jitterError = memoryPool::cuda::makeBuffer<reco::StorageScalarType>(sizeEE, deleter);
        }

        recHitsEE.did = memoryPool::cuda::makeBuffer<uint32_t>(sizeEE, deleter);
        recHitsEE.flags = memoryPool::cuda::makeBuffer<uint32_t>(sizeEE, deleter);
      }
    };

    template <typename EigenM>
    constexpr auto getLength() -> uint32_t {
      return EigenM::RowsAtCompileTime * EigenM::ColsAtCompileTime;
    }

    struct EventDataForScratchGPU {
      using SVT = SampleVector::Scalar;
      using SGVT = SampleGainVector::Scalar;
      using SMT = SampleMatrix::Scalar;
      using PMT = PulseMatrixType::Scalar;
      using BXVT = BXVectorType::Scalar;

      memoryPool::Buffer<SVT> samples;
      memoryPool::Buffer<SGVT> gainsNoise;

      memoryPool::Buffer<SMT> noisecov;
      memoryPool::Buffer<PMT> pulse_matrix;
      memoryPool::Buffer<BXVT> activeBXs;
      memoryPool::Buffer<char> acState;

      memoryPool::Buffer<bool> hasSwitchToGain6, hasSwitchToGain1, isSaturated;

      memoryPool::Buffer<SVT> sample_values, sample_value_errors;
      memoryPool::Buffer<bool> useless_sample_values;
      memoryPool::Buffer<SVT> chi2sNullHypot;
      memoryPool::Buffer<SVT> sum0sNullHypot;
      memoryPool::Buffer<SVT> sumAAsNullHypot;
      memoryPool::Buffer<char> pedestal_nums;
      memoryPool::Buffer<SVT> tMaxAlphaBetas, tMaxErrorAlphaBetas;
      memoryPool::Buffer<SVT> accTimeMax, accTimeWgt;
      memoryPool::Buffer<SVT> ampMaxAlphaBeta, ampMaxError;
      memoryPool::Buffer<SVT> timeMax, timeError;
      memoryPool::Buffer<TimeComputationState> tcState;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        constexpr auto svlength = getLength<SampleVector>();
        constexpr auto sgvlength = getLength<SampleGainVector>();
        constexpr auto smlength = getLength<SampleMatrix>();
        constexpr auto pmlength = getLength<PulseMatrixType>();
        constexpr auto bxvlength = getLength<BXVectorType>();
        auto const size = configParameters.maxNumberHitsEB + configParameters.maxNumberHitsEE;
        memoryPool::Deleter deleter =
            memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(cudaStream, memoryPool::onDevice));
        assert(deleter.pool());
        auto alloc = [&](auto& var, uint32_t size) {
          using element_type = typename std::remove_reference_t<decltype(var)>::value_type;
          var = memoryPool::cuda::makeBuffer<element_type>(size, deleter);
        };

        alloc(samples, size * svlength);
        alloc(gainsNoise, size * sgvlength);

        alloc(noisecov, size * smlength);
        alloc(pulse_matrix, size * pmlength);
        alloc(activeBXs, size * bxvlength);
        alloc(acState, size);

        alloc(hasSwitchToGain6, size);
        alloc(hasSwitchToGain1, size);
        alloc(isSaturated, size);

        if (configParameters.shouldRunTimingComputation) {
          alloc(sample_values, size * svlength);
          alloc(sample_value_errors, size * svlength);
          alloc(useless_sample_values, size * EcalDataFrame::MAXSAMPLES);
          alloc(chi2sNullHypot, size);
          alloc(sum0sNullHypot, size);
          alloc(sumAAsNullHypot, size);
          alloc(pedestal_nums, size);

          alloc(tMaxAlphaBetas, size);
          alloc(tMaxErrorAlphaBetas, size);
          alloc(accTimeMax, size);
          alloc(accTimeWgt, size);
          alloc(ampMaxAlphaBeta, size);
          alloc(ampMaxError, size);
          alloc(timeMax, size);
          alloc(timeError, size);
          alloc(tcState, size);
        }
      }
    };

    // const refs products to conditions
    struct ConditionsProducts {
      EcalPedestalsGPU::Product const& pedestals;
      EcalGainRatiosGPU::Product const& gainRatios;
      EcalPulseShapesGPU::Product const& pulseShapes;
      EcalPulseCovariancesGPU::Product const& pulseCovariances;
      EcalSamplesCorrelationGPU::Product const& samplesCorrelation;
      EcalTimeBiasCorrectionsGPU::Product const& timeBiasCorrections;
      EcalTimeCalibConstantsGPU::Product const& timeCalibConstants;
      EcalSampleMask const& sampleMask;
      EcalTimeOffsetConstant const& timeOffsetConstant;
      uint32_t offsetForHashes;
      EcalMultifitParametersGPU::Product const& multifitParameters;
    };

    struct xyz {
      int x, y, z;
    };

    struct conf_data {
      xyz threads;
      bool runV1;
      cudaStream_t cuStream;
    };

  }  // namespace multifit
}  // namespace ecal

//
// ECAL Rechit producer
//

namespace ecal {
  namespace rechit {

    // parameters that are read in the configuration file for rechit producer
    struct ConfigurationParameters {
      // device ptrs
      const int* ChannelStatusToBeExcluded = nullptr;
      uint32_t ChannelStatusToBeExcludedSize;

      bool killDeadChannels;

      bool recoverEBIsolatedChannels;
      bool recoverEEIsolatedChannels;
      bool recoverEBVFE;
      bool recoverEEVFE;
      bool recoverEBFE;
      bool recoverEEFE;

      float EBLaserMIN;
      float EELaserMIN;
      float EBLaserMAX;
      float EELaserMAX;

      const int* expanded_v_DB_reco_flags;
      const uint32_t* expanded_Sizes_v_DB_reco_flags;
      const uint32_t* expanded_flagbit_v_DB_reco_flags;
      uint32_t expanded_v_DB_reco_flagsSize;

      uint32_t flagmask;
      uint32_t maxNumberHitsEB;
      uint32_t maxNumberHitsEE;
    };

    struct EventOutputDataGPU {
      RecHit<::calo::common::DevStoragePolicy> recHitsEB, recHitsEE;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        memoryPool::Deleter deleter =
            memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(cudaStream, memoryPool::onDevice));
        assert(deleter.pool());
        //---- configParameters -> needed only to decide if to save the timing information or not
        auto const sizeEB = configParameters.maxNumberHitsEB;
        recHitsEB.energy = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEB, deleter);
        recHitsEB.time = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEB, deleter);
        recHitsEB.chi2 = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEB, deleter);
        recHitsEB.flagBits = memoryPool::cuda::makeBuffer<uint32_t>(sizeEB, deleter);
        recHitsEB.extra = memoryPool::cuda::makeBuffer<uint32_t>(sizeEB, deleter);
        recHitsEB.did = memoryPool::cuda::makeBuffer<uint32_t>(sizeEB, deleter);

        auto const sizeEE = configParameters.maxNumberHitsEE;
        recHitsEE.energy = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEE, deleter);
        recHitsEE.time = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEE, deleter);
        recHitsEE.chi2 = memoryPool::cuda::makeBuffer<::ecal::reco::StorageScalarType>(sizeEE, deleter);
        recHitsEE.flagBits = memoryPool::cuda::makeBuffer<uint32_t>(sizeEE, deleter);
        recHitsEE.extra = memoryPool::cuda::makeBuffer<uint32_t>(sizeEE, deleter);
        recHitsEE.did = memoryPool::cuda::makeBuffer<uint32_t>(sizeEE, deleter);
      }
    };

    struct EventInputDataGPU {
      ecal::UncalibratedRecHit<calo::common::DevStoragePolicy> const& ebUncalibRecHits;
      ecal::UncalibratedRecHit<calo::common::DevStoragePolicy> const& eeUncalibRecHits;
    };

    // const refs products to conditions
    struct ConditionsProducts {
      EcalRechitADCToGeVConstantGPU::Product const& ADCToGeV;
      EcalIntercalibConstantsGPU::Product const& Intercalib;
      EcalRechitChannelStatusGPU::Product const& ChannelStatus;

      EcalLaserAPDPNRatiosGPU::Product const& LaserAPDPNRatios;
      EcalLaserAPDPNRatiosRefGPU::Product const& LaserAPDPNRatiosRef;
      EcalLaserAlphasGPU::Product const& LaserAlphas;
      EcalLinearCorrectionsGPU::Product const& LinearCorrections;

      uint32_t offsetForHashes;
    };

  }  // namespace rechit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h
