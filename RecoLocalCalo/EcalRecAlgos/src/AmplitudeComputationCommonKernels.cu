#include <iostream>
#include <limits>

#include "cuda.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"

#include "AmplitudeComputationCommonKernels.h"
#include "inplace_fnnls.h"
#include "KernelHelpers.h"

namespace ecal { namespace multifit {

///
/// assume kernel launch configuration is 
/// (MAXSAMPLES * nchannels, blocks)
/// TODO: is there a point to split this kernel further to separate reductions
/// 
__global__
void kernel_prep_1d_and_initialize(
                    EcalPulseShape const* shapes_in,
                    uint16_t const* digis_in,
                    uint32_t const* dids,
                    SampleVector* amplitudes,
                    SampleVector* amplitudesForMinimization,
                    SampleGainVector* gainsNoise,
                    float const* mean_x1,
                    float const* mean_x12,
                    float const* rms_x12,
                    float const* mean_x6,
                    float const* gain6Over1,
                    float const* gain12Over6,
                    bool* hasSwitchToGain6,
                    bool* hasSwitchToGain1,
                    bool* isSaturated,
                    ::ecal::reco::StorageScalarType* energies,
                    ::ecal::reco::StorageScalarType* chi2,
                    ::ecal::reco::StorageScalarType* g_pedestal,
                    uint32_t *flags,
                    char* acState,
                    BXVectorType *bxs,
                    uint32_t const offsetForHashes,
                    bool const gainSwitchUseMaxSampleEB,
                    bool const gainSwitchUseMaxSampleEE,
                    int const nchannels) {
    constexpr bool dynamicPedestal = false;  //---- default to false, ok
    constexpr int nsamples = EcalDataFrame::MAXSAMPLES;
    constexpr int sample_max = 5;
    constexpr int full_pulse_max = 9;
    int const tx = threadIdx.x + blockIdx.x*blockDim.x;
    int const nchannels_per_block = blockDim.x / nsamples;
    int const total_threads = nchannels * nsamples;
    int const ch = tx / nsamples;
    int const sample = threadIdx.x % nsamples;

    if (ch < nchannels) {
        // array of 10 x channels per block
        // TODO: any other way of doing simple reduction
        // assume bool is 1 byte, should be quite safe
        extern __shared__ char shared_mem[];
        bool* shr_hasSwitchToGain6 = reinterpret_cast<bool*>(
            shared_mem);
        bool* shr_hasSwitchToGain1 = shr_hasSwitchToGain6 + 
            nchannels_per_block*nsamples;
        bool* shr_hasSwitchToGain0 = shr_hasSwitchToGain1 + 
            nchannels_per_block*nsamples;
        bool* shr_isSaturated = shr_hasSwitchToGain0 + 
            nchannels_per_block*nsamples;
        bool* shr_hasSwitchToGain0_tmp = shr_isSaturated + 
            nchannels_per_block*nsamples;
        char* shr_counts = reinterpret_cast<char*>(
            shr_hasSwitchToGain0_tmp) + nchannels_per_block*nsamples;

        //
        // indices
        //
        auto const did = DetId{dids[ch]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        // TODO offset for ee, 0 for eb
        auto const hashedId = isBarrel
            ? hashedIndexEB(did.rawId())
            : offsetForHashes + hashedIndexEE(did.rawId());

        //
        // pulse shape template
        /*
        for (int isample=sample; isample<EcalPulseShape::TEMPLATESAMPLES; 
            isample+=nsamples)
            shapes_out[ch](isample + 7) = shapes_in[hashedId].pdfval[isample];
            */
        
        // will be used in the future for setting state
        auto const rmsForChecking = rms_x12[hashedId];

        //
        // amplitudes
        //
        int const adc = ecal::mgpa::adc(digis_in[tx]);
        int const gainId = ecal::mgpa::gainId(digis_in[tx]);
        SampleVector::Scalar amplitude = 0.;
        SampleVector::Scalar pedestal = 0.;
        SampleVector::Scalar gainratio = 0.;

        // store into shared mem for initialization
        shr_hasSwitchToGain6[threadIdx.x] = gainId == EcalMgpaBitwiseGain6;
        shr_hasSwitchToGain1[threadIdx.x] = gainId == EcalMgpaBitwiseGain1;
        shr_hasSwitchToGain0_tmp[threadIdx.x] = gainId == EcalMgpaBitwiseGain0;
        shr_hasSwitchToGain0[threadIdx.x] = shr_hasSwitchToGain0_tmp[threadIdx.x];
        shr_counts[threadIdx.x] = 0;
        __syncthreads();
        
        // non-divergent branch (except for the last 4 threads)
        if (threadIdx.x<=blockDim.x-5) {
            #pragma unroll
            for (int i=0; i<5; i++)
                shr_counts[threadIdx.x] += 
                    shr_hasSwitchToGain0[threadIdx.x+i];
        }
        shr_isSaturated[threadIdx.x] = shr_counts[threadIdx.x] == 5;

        //
        // unrolled reductions
        // TODO
        //
        if (sample < 5) {
            shr_hasSwitchToGain6[threadIdx.x] = 
                shr_hasSwitchToGain6[threadIdx.x] ||
                shr_hasSwitchToGain6[threadIdx.x + 5];
            shr_hasSwitchToGain1[threadIdx.x] =
                shr_hasSwitchToGain1[threadIdx.x] ||
                shr_hasSwitchToGain1[threadIdx.x + 5];
            
            // duplication of hasSwitchToGain0 in order not to
            // introduce another syncthreads
            shr_hasSwitchToGain0_tmp[threadIdx.x] = 
                shr_hasSwitchToGain0_tmp[threadIdx.x] || 
                shr_hasSwitchToGain0_tmp[threadIdx.x+5];
        }
        __syncthreads();
        
        if (sample<2) {
            // note, both threads per channel take value [3] twice to avoid another if
            shr_hasSwitchToGain6[threadIdx.x] = 
                shr_hasSwitchToGain6[threadIdx.x] ||
                shr_hasSwitchToGain6[threadIdx.x+2] || 
                shr_hasSwitchToGain6[threadIdx.x+3];
            shr_hasSwitchToGain1[threadIdx.x] =
                shr_hasSwitchToGain1[threadIdx.x] ||
                shr_hasSwitchToGain1[threadIdx.x+2] || 
                shr_hasSwitchToGain1[threadIdx.x+3];

            shr_hasSwitchToGain0_tmp[threadIdx.x] = 
                shr_hasSwitchToGain0_tmp[threadIdx.x] ||
                shr_hasSwitchToGain0_tmp[threadIdx.x+2] || 
                shr_hasSwitchToGain0_tmp[threadIdx.x+3];

            // sample < 2 -> first 2 threads of each channel will be used here
            // => 0 -> will compare 3 and 4 and put into 0
            // => 1 -> will compare 4 and 5 and put into 1
            shr_isSaturated[threadIdx.x] = 
                shr_isSaturated[threadIdx.x+3] || shr_isSaturated[threadIdx.x+4];
        }
        __syncthreads();

        bool check_hasSwitchToGain0 = false;

        if (sample==0) {
            shr_hasSwitchToGain6[threadIdx.x] = 
                shr_hasSwitchToGain6[threadIdx.x] || 
                shr_hasSwitchToGain6[threadIdx.x+1];
            shr_hasSwitchToGain1[threadIdx.x] = 
                shr_hasSwitchToGain1[threadIdx.x] ||
                shr_hasSwitchToGain1[threadIdx.x+1];
            shr_hasSwitchToGain0_tmp[threadIdx.x] =
                shr_hasSwitchToGain0_tmp[threadIdx.x] ||
                shr_hasSwitchToGain0_tmp[threadIdx.x+1];

            hasSwitchToGain6[ch] = shr_hasSwitchToGain6[threadIdx.x];
            hasSwitchToGain1[ch] = shr_hasSwitchToGain1[threadIdx.x];

            // set only for the threadIdx.x corresponding to sample==0
            check_hasSwitchToGain0 = shr_hasSwitchToGain0_tmp[threadIdx.x];

            shr_isSaturated[threadIdx.x+3] = 
                shr_isSaturated[threadIdx.x] || 
                shr_isSaturated[threadIdx.x+1];
            isSaturated[ch] = shr_isSaturated[threadIdx.x+3];
        }

        // TODO: w/o this sync, there is a race
        // if (threadIdx == sample_max) below uses max sample thread, not for 0 sample
        // check if we can remove it
        __syncthreads();
        
        // TODO: divergent branch
        if (gainId==0 || gainId==3) {
            pedestal = mean_x1[hashedId];
            gainratio = gain6Over1[hashedId] * gain12Over6[hashedId];
            gainsNoise[ch](sample) = 2;
        } else if (gainId==1) {
            pedestal = mean_x12[hashedId];
            gainratio = 1.;
            gainsNoise[ch](sample) = 0;
        } else if (gainId==2) {
            pedestal = mean_x6[hashedId];
            gainratio = gain12Over6[hashedId];
            gainsNoise[ch](sample)  = 1;
        }
        
        // TODO: compile time constant -> branch should be non-divergent
        if (dynamicPedestal)
            amplitude = static_cast<SampleVector::Scalar>(adc) * gainratio;
        else
            amplitude = (static_cast<SampleVector::Scalar>(adc) - pedestal) * gainratio;
        amplitudes[ch][sample] = amplitude;

#ifdef ECAL_RECO_CUDA_DEBUG
        printf("%d %d %d %d %f %f %f\n", tx, ch, sample, adc, amplitude,
            pedestal, gainratio);
        if (adc==0)
            printf("adc is zero\n");
#endif

        //
        // initialization
        //
        amplitudesForMinimization[ch](sample) = 0;
        bxs[ch](sample) = sample - 5;

        // select the thread for the max sample 
        //---> hardcoded above to be 5th sample, ok
        if (sample == sample_max) {
            //
            // initialization
            //
            acState[ch] = static_cast<char>(MinimizationState::NotFinished);
            energies[ch] = 0;
            chi2[ch] = 0;
            g_pedestal[ch] = 0;
            uint32_t flag = 0;

            // start of this channel in shared mem
            int const chStart = threadIdx.x - sample_max;
            // thread for the max sample in shared mem
            int const threadMax = threadIdx.x;
            auto const gainSwitchUseMaxSample = isBarrel
                ? gainSwitchUseMaxSampleEB
                : gainSwitchUseMaxSampleEE;
            
            // this flag setting is applied to all of the cases
            if (shr_hasSwitchToGain6[chStart])
                flag |= 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain6;
            if (shr_hasSwitchToGain1[chStart])
                flag |= 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;

            // this corresponds to cpu branching on lastSampleBeforeSaturation
            // likely false
            if (check_hasSwitchToGain0) {
                // assign for the case some sample having gainId == 0
                //energies[ch] = amplitudes[ch][sample_max];
                energies[ch] = amplitude;

                // check if samples before sample_max have true
                bool saturated_before_max = false;
                #pragma unroll
                for (char ii=0; ii<5; ii++)
                    saturated_before_max = saturated_before_max ||
                        shr_hasSwitchToGain0[chStart + ii];

                // if saturation is in the max sample and not in the first 5
                if (!saturated_before_max && 
                    shr_hasSwitchToGain0[threadMax])
                    energies[ch] = 49140; // 4095 * 12
                    //---- AM FIXME : no pedestal subtraction???  
                    //It should be "(4095. - pedestal) * gainratio"

                // set state flag to terminate further processing of this channel
                acState[ch] = static_cast<char>(MinimizationState::Precomputed); 
                flag |= 0x1 << EcalUncalibratedRecHit::kSaturated;
                flags[ch] = flag;
                return;
            }

            // according to cpu version
//            auto max_amplitude = amplitudes[ch][sample_max]; 
            auto const max_amplitude = amplitude;
            // according to cpu version
            auto shape_value = shapes_in[hashedId].pdfval[full_pulse_max-7]; 
            // note, no syncing as the same thread will be accessing here
            bool hasGainSwitch = shr_hasSwitchToGain6[chStart]
                || shr_hasSwitchToGain1[chStart]
                || shr_isSaturated[chStart+3];

            // pedestal is final unconditionally
            g_pedestal[ch] = pedestal;
            if (hasGainSwitch && gainSwitchUseMaxSample) {
                // thread for sample=0 will access the right guys
                energies[ch] = max_amplitude / shape_value;
                acState[ch] = static_cast<char>(MinimizationState::Precomputed);
                flags[ch] = flag;
                return;
            }
            
            // this happens cause sometimes rms_x12 is 0...
            // needs to be checkec why this is the case
            // general case here is that noisecov is a Zero matrix
            if (rmsForChecking == 0) {
                acState[ch] = static_cast<char>(MinimizationState::Precomputed);
                flags[ch] = flag;
                return;
            }

            // for the case when no shortcuts were taken
            flags[ch] = flag;
        }
    }
}

///
/// assume kernel launch configuration is 
/// ([MAXSAMPLES, MAXSAMPLES], nchannels)
///
__global__
void kernel_prep_2d(EcalPulseCovariance const* pulse_cov_in,
                    FullSampleMatrix* pulse_cov_out,
                    SampleGainVector const* gainNoise,
                    uint32_t const* dids,
                    float const* rms_x12,
                    float const* rms_x6,
                    float const* rms_x1,
                    float const* gain12Over6,
                    float const* gain6Over1,
                    double const* G12SamplesCorrelationEB,
                    double const* G6SamplesCorrelationEB,
                    double const* G1SamplesCorrelationEB,
                    double const* G12SamplesCorrelationEE,
                    double const* G6SamplesCorrelationEE,
                    double const* G1SamplesCorrelationEE,
                    SampleMatrix* noisecov,
                    PulseMatrixType* pulse_matrix,
                    EcalPulseShape const* pulse_shape,
                    bool const* hasSwitchToGain6,
                    bool const* hasSwitchToGain1,
                    bool const* isSaturated,
                    uint32_t const offsetForHashes) {
    int ch = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    constexpr int nsamples = EcalDataFrame::MAXSAMPLES;
    constexpr float addPedestalUncertainty = 0.f;
    constexpr bool dynamicPedestal = false;
    constexpr bool simplifiedNoiseModelForGainSwitch = true;  //---- default is true
    constexpr int template_samples = EcalPulseShape::TEMPLATESAMPLES;

    bool tmp0 = hasSwitchToGain6[ch];
    bool tmp1 = hasSwitchToGain1[ch];
    auto const did = DetId{dids[ch]};
    auto const isBarrel = did.subdetId() == EcalBarrel;
    auto const hashedId = isBarrel
        ? hashedIndexEB(did.rawId())
        : offsetForHashes + hashedIndexEE(did.rawId());
    auto const G12SamplesCorrelation = isBarrel
        ? G12SamplesCorrelationEB
        : G12SamplesCorrelationEE;
    auto const* G6SamplesCorrelation = isBarrel
        ? G6SamplesCorrelationEB
        : G6SamplesCorrelationEE;
    auto const* G1SamplesCorrelation = isBarrel
        ? G1SamplesCorrelationEB
        : G1SamplesCorrelationEE;
    bool tmp2 = isSaturated[ch];
    bool hasGainSwitch = tmp0 || tmp1 || tmp2;
    auto const vidx = ecal::abs(ty - tx);

    // only ty == 0 and 1 will go for a second iteration
    for (int iy=ty; iy<template_samples; iy+=nsamples)
        for (int ix=tx; ix<template_samples; ix+=nsamples)
            pulse_cov_out[ch](iy+7, ix+7) = pulse_cov_in[hashedId].covval[iy][ix];

    // non-divergent branch for all threads per block
    if (hasGainSwitch) {
        // TODO: did not include simplified noise model
        float noise_value = 0;

        // non-divergent branch - all threads per block
        // TODO: all of these constants indicate that 
        // that these parts could be splitted into completely different 
        // kernels and run one of them only depending on the config
        if (simplifiedNoiseModelForGainSwitch) {
            int isample_max = 5; // according to cpu defs
            int gainidx = gainNoise[ch][isample_max];

            // non-divergent branches
            if (gainidx==0)
                //noise_value = rms_x12[ch]*rms_x12[ch]*noisecorrs[0](ty, tx);
                noise_value = rms_x12[hashedId]*rms_x12[hashedId]
                    * G12SamplesCorrelation[vidx];
            if (gainidx==1) 
//                noise_value = gain12Over6[ch]*gain12Over6[ch] * rms_x6[ch]*rms_x6[ch]
//                    *noisecorrs[1](ty, tx);
                noise_value = gain12Over6[hashedId]*gain12Over6[hashedId] 
                    * rms_x6[hashedId]*rms_x6[hashedId]
                    * G6SamplesCorrelation[vidx];
            if (gainidx==2)
//                noise_value = gain12Over6[ch]*gain12Over6[ch]
//                    * gain6Over1[ch]*gain6Over1[ch] * rms_x1[ch]*rms_x1[ch]
//                    * noisecorrs[2](ty, tx);
                noise_value = gain12Over6[hashedId]*gain12Over6[hashedId]
                    * gain6Over1[hashedId]*gain6Over1[hashedId] 
                    * rms_x1[hashedId]*rms_x1[hashedId]
                    * G1SamplesCorrelation[vidx];
            if (!dynamicPedestal && addPedestalUncertainty>0.f)
                noise_value += addPedestalUncertainty*addPedestalUncertainty;
        } else {
            int gainidx=0;
            char mask = gainidx;
            int pedestal = gainNoise[ch][ty] == mask ? 1 : 0;
//            noise_value += /* gainratio is 1*/ rms_x12[ch]*rms_x12[ch]
//                *pedestal*noisecorrs[0](ty, tx);
            noise_value += /* gainratio is 1*/ rms_x12[hashedId]*rms_x12[hashedId]
                * pedestal* G12SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty>0.f) {
                noise_value += /* gainratio is 1 */
                    addPedestalUncertainty*addPedestalUncertainty*pedestal;
            }

            //
            gainidx=1;
            mask = gainidx;
            pedestal = gainNoise[ch][ty] == mask ? 1 : 0;
//            noise_value += gain12Over6[ch]*gain12Over6[ch]
//                *rms_x6[ch]*rms_x6[ch]*pedestal*noisecorrs[1](ty, tx);
            noise_value += gain12Over6[hashedId]*gain12Over6[hashedId]
                *rms_x6[hashedId]*rms_x6[hashedId]*pedestal
                * G6SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty>0.f) {
                noise_value += gain12Over6[hashedId]*gain12Over6[hashedId]
                    *addPedestalUncertainty*addPedestalUncertainty
                    *pedestal;
            }
            
            //
            gainidx=2;
            mask = gainidx;
            pedestal = gainNoise[ch][ty] == mask ? 1 : 0;
            float tmp = gain6Over1[hashedId] * gain12Over6[hashedId];
//            noise_value += tmp*tmp * rms_x1[ch]*rms_x1[ch]
//                *pedestal*noisecorrs[2](ty, tx);
            noise_value += tmp*tmp * rms_x1[hashedId]*rms_x1[hashedId]
                *pedestal* G1SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty>0.f) {
                noise_value += tmp*tmp * addPedestalUncertainty*addPedestalUncertainty
                    * pedestal;
            }
        }

        noisecov[ch](ty, tx) = noise_value;
    } else {
        auto rms = rms_x12[hashedId];
        float noise_value = rms*rms * G12SamplesCorrelation[vidx];
        if (!dynamicPedestal && addPedestalUncertainty>0.f) {
            //----  add fully correlated component to noise covariance to inflate pedestal uncertainty
            noise_value += addPedestalUncertainty*addPedestalUncertainty;
        }
        noisecov[ch](ty, tx) = noise_value;
    }

    // pulse matrix
//    int const bx = tx - 5; // -5 -4 -3 ... 3 4
//    int bx = (*bxs)(tx);
//    int const offset = 7 - 3 - bx;
    int const posToAccess = 9 - tx + ty; // see cpu for reference
    float const value = posToAccess>=7 
        ? pulse_shape[hashedId].pdfval[posToAccess-7]
        : 0;
    pulse_matrix[ch](ty, tx) = value;
}

__global__
void kernel_permute_results(
        SampleVector *amplitudes,
        BXVectorType const*activeBXs,
        ::ecal::reco::StorageScalarType *energies,
        char const* acState,
        int const nchannels) {
    // constants
    constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

    // indices
    int const tx = threadIdx.x + blockIdx.x * blockDim.x;
    int const ch = tx / nsamples;
    int const iii = tx % nsamples; // this is to address activeBXs

    if (ch >= nchannels) return;
    
    // channels that have amplitude precomputed do not need results to be permuted
    auto const state = static_cast<MinimizationState>(acState[ch]);
    if (static_cast<MinimizationState>(acState[ch]) ==
        MinimizationState::Precomputed)
        return;

    // configure shared memory and cp into it
    extern __shared__ char smem[];
    SampleVector::Scalar* values = reinterpret_cast<SampleVector::Scalar*>(
        smem);
    values[threadIdx.x] = amplitudes[ch](iii);
    __syncthreads();

    // get the sample for this bx
    auto const sample = static_cast<int>(activeBXs[ch](iii)) + 5;

    // store back to global
    amplitudes[ch](sample) = values[threadIdx.x];

    // store sample 5 separately
    // only for the case when minimization was performed
    // not for cases with precomputed amplitudes
    if (sample == 5)
        energies[ch] = values[threadIdx.x];
}

///
/// Build an Ecal RecHit.
/// TODO: Use SoA data structures on the host directly
/// the reason for removing this from minimize kernel is to isolate the minimize + 
/// again, building an aos rec hit involves strides... -> bad memory access pattern
///
#ifdef RUN_BUILD_AOS_RECHIT
__global__
void kernel_build_rechit(
    float const* energies,
    float const* chi2s,
    uint32_t* dids,
    EcalUncalibratedRecHit* rechits,
    int nchannels) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nchannels) {
        rechits[idx] = EcalUncalibratedRecHit{dids[idx], energies[idx],
            0, 0, chi2s[idx], 0};
    }
}
#endif

}}
