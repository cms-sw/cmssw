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
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalGainRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalIntercalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosRefGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAlphasGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLinearCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPedestalsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseCovariancesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseShapesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitADCToGeVConstantGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitChannelStatusGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSamplesCorrelationGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeBiasCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeCalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalMultifitParametersGPU.h"

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
      type *amplitudeFitParametersEB = nullptr, *amplitudeFitParametersEE = nullptr;

      uint32_t timeFitParametersSizeEB, timeFitParametersSizeEE;
      // device ptrs
      type *timeFitParametersEB = nullptr, *timeFitParametersEE = nullptr;

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

      uint32_t maxNumberHits;
    };

    struct EventOutputDataGPU {
      UncalibratedRecHit<::calo::common::DevStoragePolicy> recHitsEB, recHitsEE;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        auto const size = configParameters.maxNumberHits;
        recHitsEB.amplitudesAll = cms::cuda::make_device_unique<reco::ComputationScalarType[]>(size * EcalDataFrame::MAXSAMPLES, cudaStream);
        recHitsEB.amplitude = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHitsEB.chi2 = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHitsEB.pedestal = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);

        if (configParameters.shouldRunTimingComputation) {
          recHitsEB.jitter = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
          recHitsEB.jitterError = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        }

        recHitsEB.did = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEB.flags = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        
        recHitsEE.amplitudesAll = cms::cuda::make_device_unique<reco::ComputationScalarType[]>(size * EcalDataFrame::MAXSAMPLES, cudaStream);
        recHitsEE.amplitude = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHitsEE.chi2 = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        recHitsEE.pedestal = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);

        if (configParameters.shouldRunTimingComputation) {
          recHitsEE.jitter = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
          recHitsEE.jitterError = cms::cuda::make_device_unique<reco::StorageScalarType[]>(size, cudaStream);
        }

        recHitsEE.did = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEE.flags = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
      }
    };
    
    template<typename EigenM>
    constexpr auto getLength() -> uint32_t {
      return EigenM::RowsAtCompileTime * EigenM::ColsAtCompileTime;
    }

    struct EventDataForScratchGPU {
      using SVT = SampleVector::Scalar;
      using SGVT = SampleGainVector::Scalar;
      using SMT = SampleMatrix::Scalar;
      using PMT = PulseMatrixType::Scalar;
      using BXVT = BXVectorType::Scalar;

      cms::cuda::device::unique_ptr<SVT[]> samples;
      cms::cuda::device::unique_ptr<SGVT[]> gainsNoise;

      cms::cuda::device::unique_ptr<SMT[]> noisecov;
      cms::cuda::device::unique_ptr<PMT[]> pulse_matrix;
      cms::cuda::device::unique_ptr<BXVT[]> activeBXs;
      cms::cuda::device::unique_ptr<char[]> acState;

      cms::cuda::device::unique_ptr<bool[]> hasSwitchToGain6, hasSwitchToGain1, isSaturated;

      cms::cuda::device::unique_ptr<SVT[]> sample_values, sample_value_errors;
      cms::cuda::device::unique_ptr<bool[]> useless_sample_values;
      cms::cuda::device::unique_ptr<SVT[]> chi2sNullHypot;
      cms::cuda::device::unique_ptr<SVT[]> sum0sNullHypot;
      cms::cuda::device::unique_ptr<SVT[]> sumAAsNullHypot;
      cms::cuda::device::unique_ptr<char[]> pedestal_nums;
      cms::cuda::device::unique_ptr<SVT[]> tMaxAlphaBetas, tMaxErrorAlphaBetas;
      cms::cuda::device::unique_ptr<SVT[]> accTimeMax, accTimeWgt;
      cms::cuda::device::unique_ptr<SVT[]> ampMaxAlphaBeta, ampMaxError;
      cms::cuda::device::unique_ptr<SVT[]> timeMax, timeError;
      cms::cuda::device::unique_ptr<TimeComputationState[]> tcState;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        constexpr auto svlength = getLength<SampleVector>();
        constexpr auto sgvlength = getLength<SampleGainVector>();
        constexpr auto smlength = getLength<SampleMatrix>();
        constexpr auto pmlength = getLength<PulseMatrixType>();
        constexpr auto bxvlength = getLength<BXVectorType>();
        auto const size = configParameters.maxNumberHits;

#define MYMALLOC(var, size) var = cms::cuda::make_device_unique<decltype(var)::element_type[]>(size, cudaStream)
        MYMALLOC(samples, size*svlength);
        //cudaCheck(cudaMalloc((void**)&samples, size * sizeof(SampleVector)));
        MYMALLOC(gainsNoise, size*sgvlength);
        //cudaCheck(cudaMalloc((void**)&gainsNoise, size * sizeof(SampleGainVector)));

        MYMALLOC(noisecov, size * smlength);
        //cudaCheck(cudaMalloc((void**)&noisecov, size * sizeof(SampleMatrix)));
        MYMALLOC(pulse_matrix, size * pmlength);
        //cudaCheck(cudaMalloc((void**)&pulse_matrix, size * sizeof(PulseMatrixType)));
        MYMALLOC(activeBXs, size*bxvlength);
        //cudaCheck(cudaMalloc((void**)&activeBXs, size * sizeof(BXVectorType)));
        MYMALLOC(acState, size);
        //cudaCheck(cudaMalloc((void**)&acState, size * sizeof(char)));

        MYMALLOC(hasSwitchToGain6, size);
        //cudaCheck(cudaMalloc((void**)&hasSwitchToGain6, size * sizeof(bool)));
        MYMALLOC(hasSwitchToGain1, size);
        //cudaCheck(cudaMalloc((void**)&hasSwitchToGain1, size * sizeof(bool)));
        MYMALLOC(isSaturated, size);
        //cudaCheck(cudaMalloc((void**)&isSaturated, size * sizeof(bool)));

        if (configParameters.shouldRunTimingComputation) {
          MYMALLOC(sample_values, size*svlength);
          //cudaCheck(cudaMalloc((void**)&sample_values, size * sizeof(SampleVector)));
          MYMALLOC(sample_value_errors, size*svlength);
          //cudaCheck(cudaMalloc((void**)&sample_value_errors, size * sizeof(SampleVector)));
          MYMALLOC(useless_sample_values, size*EcalDataFrame::MAXSAMPLES);
          //cudaCheck(cudaMalloc((void**)&useless_sample_values, size * sizeof(bool) * EcalDataFrame::MAXSAMPLES));
          MYMALLOC(chi2sNullHypot, size);
          //cudaCheck(cudaMalloc((void**)&chi2sNullHypot, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(sum0sNullHypot, size);
          //cudaCheck(cudaMalloc((void**)&sum0sNullHypot, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(sumAAsNullHypot, size);
          //cudaCheck(cudaMalloc((void**)&sumAAsNullHypot, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(pedestal_nums, size);
          //cudaCheck(cudaMalloc((void**)&pedestal_nums, size * sizeof(char)));

          MYMALLOC(tMaxAlphaBetas, size);
          //cudaCheck(cudaMalloc((void**)&tMaxAlphaBetas, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(tMaxErrorAlphaBetas, size);
          //cudaCheck(cudaMalloc((void**)&tMaxErrorAlphaBetas, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(accTimeMax, size);
          //cudaCheck(cudaMalloc((void**)&accTimeMax, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(accTimeWgt, size);
          //cudaCheck(cudaMalloc((void**)&accTimeWgt, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(ampMaxAlphaBeta, size);
          //cudaCheck(cudaMalloc((void**)&ampMaxAlphaBeta, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(ampMaxError, size);
          //cudaCheck(cudaMalloc((void**)&ampMaxError, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(timeMax, size);
          //cudaCheck(cudaMalloc((void**)&timeMax, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(timeError, size);
          //cudaCheck(cudaMalloc((void**)&timeError, size * sizeof(SampleVector::Scalar)));
          MYMALLOC(tcState, size);
          //cudaCheck(cudaMalloc((void**)&tcState, size * sizeof(TimeComputationState)));
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

    //*/

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
      int* ChannelStatusToBeExcluded = nullptr;
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

      //       std::vector<std::vector<uint32_t> > v_DB_reco_flags;
      int* expanded_v_DB_reco_flags;
      uint32_t* expanded_Sizes_v_DB_reco_flags;
      uint32_t* expanded_flagbit_v_DB_reco_flags;
      uint32_t expanded_v_DB_reco_flagsSize;

      uint32_t flagmask;
      uint32_t maxNumberHits;

      //
      //       bool shouldRunTimingComputation;
    };

    struct EventOutputDataGPU {
      RecHit<::calo::common::DevStoragePolicy> recHitsEB, recHitsEE;

      void allocate(ConfigurationParameters const& configParameters, cudaStream_t cudaStream) {
        //      void allocate(uint32_t size) {
        //---- configParameters -> needed only to decide if to save the timing information or not
        auto const size = configParameters.maxNumberHits;
        recHitsEB.energy = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEB.time = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEB.chi2 = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEB.flagBits = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEB.extra = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEB.did = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);

        recHitsEE.energy = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEE.time = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEE.chi2 = cms::cuda::make_device_unique<::ecal::reco::StorageScalarType[]>(size, cudaStream);
        recHitsEE.flagBits = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEE.extra = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
        recHitsEE.did = cms::cuda::make_device_unique<uint32_t[]>(size, cudaStream);
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
      //
      EcalLaserAPDPNRatiosGPU::Product const& LaserAPDPNRatios;
      EcalLaserAPDPNRatiosRefGPU::Product const& LaserAPDPNRatiosRef;
      EcalLaserAlphasGPU::Product const& LaserAlphas;
      EcalLinearCorrectionsGPU::Product const& LinearCorrections;
      //
      //
      uint32_t offsetForHashes;
    };

  }  // namespace rechit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_DeclsForKernels_h
