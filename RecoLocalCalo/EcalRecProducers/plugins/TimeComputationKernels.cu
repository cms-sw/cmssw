#include <cmath>
#include <limits>

#include <cuda.h>

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "Common.h"
#include "TimeComputationKernels.h"
#include "KernelHelpers.h"

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal {
  namespace multifit {

    __device__ __forceinline__ bool use_sample(unsigned int sample_mask, unsigned int sample) {
      return sample_mask & (0x1 << (EcalDataFrame::MAXSAMPLES - (sample + 1)));
    }

    __global__ void kernel_time_compute_nullhypot(SampleVector::Scalar const* sample_values,
                                                  SampleVector::Scalar const* sample_value_errors,
                                                  bool const* useless_sample_values,
                                                  SampleVector::Scalar* chi2s,
                                                  SampleVector::Scalar* sum0s,
                                                  SampleVector::Scalar* sumAAs,
                                                  const int nchannels) {
      using ScalarType = SampleVector::Scalar;
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      int tx = threadIdx.x + blockDim.x * blockIdx.x;
      int ltx = threadIdx.x;
      int ch = tx / nsamples;
      int nchannels_per_block = blockDim.x / nsamples;

      // threads that return here should not affect the __syncthreads() below since they have exitted the kernel
      if (ch >= nchannels)
        return;

      int sample = tx % nsamples;

      // shared mem inits
      extern __shared__ char sdata[];
      char* s_sum0 = sdata;
      SampleVector::Scalar* s_sum1 = reinterpret_cast<SampleVector::Scalar*>(s_sum0 + nchannels_per_block * nsamples);
      SampleVector::Scalar* s_sumA = s_sum1 + nchannels_per_block * nsamples;
      SampleVector::Scalar* s_sumAA = s_sumA + nchannels_per_block * nsamples;

      // TODO make sure no div by 0
      const auto inv_error =
          useless_sample_values[tx] ? 0.0 : 1.0 / (sample_value_errors[tx] * sample_value_errors[tx]);
      const auto sample_value = sample_values[tx];
      s_sum0[ltx] = useless_sample_values[tx] ? 0 : 1;
      s_sum1[ltx] = inv_error;
      s_sumA[ltx] = sample_value * inv_error;
      s_sumAA[ltx] = sample_value * sample_value * inv_error;
      __syncthreads();

      // 5 threads for [0, 4] samples
      if (sample < 5) {
        s_sum0[ltx] += s_sum0[ltx + 5];
        s_sum1[ltx] += s_sum1[ltx + 5];
        s_sumA[ltx] += s_sumA[ltx + 5];
        s_sumAA[ltx] += s_sumAA[ltx + 5];
      }
      __syncthreads();

      if (sample < 2) {
        // note double counting of sample 3
        s_sum0[ltx] += s_sum0[ltx + 2] + s_sum0[ltx + 3];
        s_sum1[ltx] += s_sum1[ltx + 2] + s_sum1[ltx + 3];
        s_sumA[ltx] += s_sumA[ltx + 2] + s_sumA[ltx + 3];
        s_sumAA[ltx] += s_sumAA[ltx + 2] + s_sumAA[ltx + 3];
      }
      __syncthreads();

      if (sample == 0) {
        // note, subtract to remove the double counting of sample == 3
        const auto sum0 = s_sum0[ltx] + s_sum0[ltx + 1] - s_sum0[ltx + 3];
        const auto sum1 = s_sum1[ltx] + s_sum1[ltx + 1] - s_sum1[ltx + 3];
        const auto sumA = s_sumA[ltx] + s_sumA[ltx + 1] - s_sumA[ltx + 3];
        const auto sumAA = s_sumAA[ltx] + s_sumAA[ltx + 1] - s_sumAA[ltx + 3];
        const auto chi2 = sum0 > 0 ? (sumAA - sumA * sumA / sum1) / sum0 : static_cast<ScalarType>(0);
        chi2s[ch] = chi2;
        sum0s[ch] = sum0;
        sumAAs[ch] = sumAA;

#ifdef DEBUG_TC_NULLHYPOT
        if (ch == 0) {
          printf("chi2 = %f sum0 = %d sumAA = %f\n", chi2, static_cast<int>(sum0), sumAA);
        }
#endif
      }
    }

    constexpr float fast_expf(float x) { return unsafe_expf<6>(x); }
    constexpr float fast_logf(float x) { return unsafe_logf<7>(x); }

    //#define DEBUG_TC_MAKERATIO
    //
    // launch ctx parameters are
    // 45 threads per channel, X channels per block, Y blocks
    // 45 comes from: 10 samples for i <- 0 to 9 and for j <- i+1 to 9
    // TODO: it might be much beter to use 32 threads per channel instead of 45
    // to simplify the synchronization
    //
    __global__ void kernel_time_compute_makeratio(SampleVector::Scalar const* sample_values,
                                                  SampleVector::Scalar const* sample_value_errors,
                                                  uint32_t const* dids_eb,
                                                  uint32_t const* dids_ee,
                                                  bool const* useless_sample_values,
                                                  char const* pedestal_nums,
                                                  ConfigurationParameters::type const* amplitudeFitParametersEB,
                                                  ConfigurationParameters::type const* amplitudeFitParametersEE,
                                                  ConfigurationParameters::type const* timeFitParametersEB,
                                                  ConfigurationParameters::type const* timeFitParametersEE,
                                                  SampleVector::Scalar const* sumAAsNullHypot,
                                                  SampleVector::Scalar const* sum0sNullHypot,
                                                  SampleVector::Scalar* tMaxAlphaBetas,
                                                  SampleVector::Scalar* tMaxErrorAlphaBetas,
                                                  SampleVector::Scalar* g_accTimeMax,
                                                  SampleVector::Scalar* g_accTimeWgt,
                                                  TimeComputationState* g_state,
                                                  unsigned const int timeFitParameters_sizeEB,
                                                  unsigned const int timeFitParameters_sizeEE,
                                                  ConfigurationParameters::type const timeFitLimits_firstEB,
                                                  ConfigurationParameters::type const timeFitLimits_firstEE,
                                                  ConfigurationParameters::type const timeFitLimits_secondEB,
                                                  ConfigurationParameters::type const timeFitLimits_secondEE,
                                                  const int nchannels,
                                                  uint32_t const offsetForInputs) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr int nthreads_per_channel = 45;  // n=10, n(n-1)/2
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int gtx = threadIdx.x + blockDim.x * blockIdx.x;
      const int ch = gtx / nthreads_per_channel;
      const int ltx = threadIdx.x % nthreads_per_channel;
      const int ch_start = ch * nsamples;
      const auto* dids = ch >= offsetForInputs ? dids_ee : dids_eb;
      const int inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;

      // remove inactive threads
      // threads that return here should not affect the __syncthreads() below since they have exitted the kernel
      if (ch >= nchannels)
        return;

      const auto did = DetId{dids[inputCh]};
      const auto isBarrel = did.subdetId() == EcalBarrel;
      const auto* amplitudeFitParameters = isBarrel ? amplitudeFitParametersEB : amplitudeFitParametersEE;
      const auto* timeFitParameters = isBarrel ? timeFitParametersEB : timeFitParametersEE;
      const auto timeFitParameters_size = isBarrel ? timeFitParameters_sizeEB : timeFitParameters_sizeEE;
      const auto timeFitLimits_first = isBarrel ? timeFitLimits_firstEB : timeFitLimits_firstEE;
      const auto timeFitLimits_second = isBarrel ? timeFitLimits_secondEB : timeFitLimits_secondEE;

      extern __shared__ char smem[];
      ScalarType* shr_chi2s = reinterpret_cast<ScalarType*>(smem);
      ScalarType* shr_time_wgt = shr_chi2s + blockDim.x;
      ScalarType* shr_time_max = shr_time_wgt + blockDim.x;
      ScalarType* shrTimeMax = shr_time_max + blockDim.x;
      ScalarType* shrTimeWgt = shrTimeMax + blockDim.x;

      // map tx -> (sample_i, sample_j)
      int sample_i, sample_j = 0;
      if (ltx >= 0 && ltx <= 8) {
        sample_i = 0;
        sample_j = 1 + ltx;
      } else if (ltx <= 16) {
        sample_i = 1;
        sample_j = 2 + ltx - 9;
      } else if (ltx <= 23) {
        sample_i = 2;
        sample_j = 3 + ltx - 17;
      } else if (ltx <= 29) {
        sample_i = 3;
        sample_j = 4 + ltx - 24;
      } else if (ltx <= 34) {
        sample_i = 4;
        sample_j = 5 + ltx - 30;
      } else if (ltx <= 38) {
        sample_i = 5;
        sample_j = 6 + ltx - 35;
      } else if (ltx <= 41) {
        sample_i = 6;
        sample_j = 7 + ltx - 39;
      } else if (ltx <= 43) {
        sample_i = 7;
        sample_j = 8 + ltx - 42;
      } else if (ltx <= 44) {
        sample_i = 8;
        sample_j = 9;
      } else
        assert(false);

      const auto tx_i = ch_start + sample_i;
      const auto tx_j = ch_start + sample_j;

      //
      // note, given the way we partition the block, with 45 threads per channel
      // we will end up with inactive threads which need to be dragged along
      // through the synching point
      //
      bool const condForUselessSamples = useless_sample_values[tx_i] || useless_sample_values[tx_j] ||
                                         sample_values[tx_i] <= 1 || sample_values[tx_j] <= 1;

      //
      // see cpu implementation for explanation
      //
      ScalarType chi2 = std::numeric_limits<ScalarType>::max();
      ScalarType tmax = 0;
      ScalarType tmaxerr = 0;
      shrTimeMax[threadIdx.x] = 0;
      shrTimeWgt[threadIdx.x] = 0;
      bool internalCondForSkipping1 = true;
      bool internalCondForSkipping2 = true;
      if (!condForUselessSamples) {
        const auto rtmp = sample_values[tx_i] / sample_values[tx_j];
        const auto invampl_i = 1.0 / sample_values[tx_i];
        const auto relErr2_i = sample_value_errors[tx_i] * sample_value_errors[tx_i] * invampl_i * invampl_i;
        const auto invampl_j = 1.0 / sample_values[tx_j];
        const auto relErr2_j = sample_value_errors[tx_j] * sample_value_errors[tx_j] * invampl_j * invampl_j;
        const auto err1 = rtmp * rtmp * (relErr2_i + relErr2_j);
        auto err2 = sample_value_errors[tx_j] * (sample_values[tx_i] - sample_values[tx_j]) * (invampl_j * invampl_j);
        // TODO non-divergent branch for a block if each block has 1 channel
        // otherwise non-divergent for groups of 45 threads
        // at this point, pedestal_nums[ch] can be either 0, 1 or 2
        if (pedestal_nums[ch] == 2)
          err2 *= err2 * 0.5;
        const auto err3 = (0.289 * 0.289) * (invampl_j * invampl_j);
        const auto total_error = std::sqrt(err1 + err2 + err3);

        const auto alpha = amplitudeFitParameters[0];
        const auto beta = amplitudeFitParameters[1];
        const auto alphabeta = alpha * beta;
        const auto invalphabeta = 1.0 / alphabeta;

        // variables instead of a struct
        const auto ratio_index = sample_i;
        const auto ratio_step = sample_j - sample_i;
        const auto ratio_value = rtmp;
        const auto ratio_error = total_error;

        const auto rlim_i_j = fast_expf(static_cast<ScalarType>(sample_j - sample_i) / beta) - 0.001;
        internalCondForSkipping1 = !(total_error < 1.0 && rtmp > 0.001 && rtmp < rlim_i_j);
        if (!internalCondForSkipping1) {
          //
          // precompute.
          // in cpu version this was done conditionally
          // however easier to do it here (precompute) and then just filter out
          // if not needed
          //
          const auto l_timeFitLimits_first = timeFitLimits_first;
          const auto l_timeFitLimits_second = timeFitLimits_second;
          if (ratio_step == 1 && ratio_value >= l_timeFitLimits_first && ratio_value <= l_timeFitLimits_second) {
            const auto time_max_i = static_cast<ScalarType>(ratio_index);
            auto u = timeFitParameters[timeFitParameters_size - 1];
            CMS_UNROLL_LOOP
            for (int k = timeFitParameters_size - 2; k >= 0; k--)
              u = u * ratio_value + timeFitParameters[k];

            auto du = (timeFitParameters_size - 1) * (timeFitParameters[timeFitParameters_size - 1]);
            for (int k = timeFitParameters_size - 2; k >= 1; k--)
              du = du * ratio_value + k * timeFitParameters[k];

            const auto error2 = ratio_error * ratio_error * du * du;
            const auto time_max = error2 > 0 ? (time_max_i - u) / error2 : static_cast<ScalarType>(0);
            const auto time_wgt = error2 > 0 ? 1.0 / error2 : static_cast<ScalarType>(0);

            // store into shared mem
            // note, this name is essentially identical to the one used
            // below.
            shrTimeMax[threadIdx.x] = error2 > 0 ? time_max : 0;
            shrTimeWgt[threadIdx.x] = error2 > 0 ? time_wgt : 0;
          } else {
            shrTimeMax[threadIdx.x] = 0;
            shrTimeWgt[threadIdx.x] = 0;
          }

          // continue with ratios
          const auto stepOverBeta = static_cast<SampleVector::Scalar>(ratio_step) / beta;
          const auto offset = static_cast<SampleVector::Scalar>(ratio_index) + alphabeta;
          const auto rmin = std::max(ratio_value - ratio_error, 0.001);
          const auto rmax = std::min(ratio_value + ratio_error,
                                     fast_expf(static_cast<SampleVector::Scalar>(ratio_step) / beta) - 0.001);
          const auto time1 = offset - ratio_step / (fast_expf((stepOverBeta - fast_logf(rmin)) / alpha) - 1.0);
          const auto time2 = offset - ratio_step / (fast_expf((stepOverBeta - fast_logf(rmax)) / alpha) - 1.0);

          // set these guys
          tmax = 0.5 * (time1 + time2);
          tmaxerr = 0.5 * std::sqrt((time1 - time2) * (time1 - time2));
#ifdef DEBUG_TC_MAKERATIO
          if (ch == 1 || ch == 0)
            printf("ch = %d ltx = %d tmax = %f tmaxerr = %f time1 = %f time2 = %f offset = %f rmin = %f rmax = %f\n",
                   ch,
                   ltx,
                   tmax,
                   tmaxerr,
                   time1,
                   time2,
                   offset,
                   rmin,
                   rmax);
#endif

          SampleVector::Scalar sumAf = 0;
          SampleVector::Scalar sumff = 0;
          const int itmin = std::max(-1, static_cast<int>(std::floor(tmax - alphabeta)));
          auto loffset = (static_cast<ScalarType>(itmin) - tmax) * invalphabeta;
          // TODO: data dependence
          for (int it = itmin + 1; it < nsamples; it++) {
            loffset += invalphabeta;
            if (useless_sample_values[ch_start + it])
              continue;
            const auto inverr2 = 1.0 / (sample_value_errors[ch_start + it] * sample_value_errors[ch_start + it]);
            const auto term1 = 1.0 + loffset;
            const auto f = (term1 > 1e-6) ? fast_expf(alpha * (fast_logf(term1) - loffset)) : 0;
            sumAf += sample_values[ch_start + it] * (f * inverr2);
            sumff += f * (f * inverr2);
          }

          const auto sumAA = sumAAsNullHypot[ch];
          const auto sum0 = sum0sNullHypot[ch];
          chi2 = sumAA;
          // TODO: sum0 can not be 0 below, need to introduce the check upfront
          if (sumff > 0) {
            chi2 = sumAA - sumAf * (sumAf / sumff);
          }
          chi2 /= sum0;

#ifdef DEBUG_TC_MAKERATIO
          if (ch == 1 || ch == 0)
            printf("ch = %d ltx = %d sumAf = %f sumff = %f sumAA = %f sum0 = %d tmax = %f tmaxerr = %f chi2 = %f\n",
                   ch,
                   ltx,
                   sumAf,
                   sumff,
                   sumAA,
                   static_cast<int>(sum0),
                   tmax,
                   tmaxerr,
                   chi2);
#endif

          if (chi2 > 0 && tmax > 0 && tmaxerr > 0)
            internalCondForSkipping2 = false;
          else
            chi2 = std::numeric_limits<ScalarType>::max();
        }
      }

      // store into smem
      shr_chi2s[threadIdx.x] = chi2;
      __syncthreads();

      // find min chi2 - quite crude for now
      // TODO validate/check
      char iter = nthreads_per_channel / 2 + nthreads_per_channel % 2;
      bool oddElements = nthreads_per_channel % 2;
      CMS_UNROLL_LOOP
      while (iter >= 1) {
        if (ltx < iter)
          // for odd ns, the last guy will just store itself
          // exception is for ltx == 0 and iter==1
          shr_chi2s[threadIdx.x] = oddElements && (ltx == iter - 1 && ltx > 0)
                                       ? shr_chi2s[threadIdx.x]
                                       : std::min(shr_chi2s[threadIdx.x], shr_chi2s[threadIdx.x + iter]);
        __syncthreads();
        oddElements = iter % 2;
        iter = iter == 1 ? iter / 2 : iter / 2 + iter % 2;
      }

      // filter out inactive or useless samples threads
      if (!condForUselessSamples && !internalCondForSkipping1 && !internalCondForSkipping2) {
        // min chi2, now compute weighted average of tmax measurements
        // see cpu version for more explanation
        const auto chi2min = shr_chi2s[threadIdx.x - ltx];
        const auto chi2Limit = chi2min + 1.0;
        const auto inverseSigmaSquared = chi2 < chi2Limit ? 1.0 / (tmaxerr * tmaxerr) : 0.0;

#ifdef DEBUG_TC_MAKERATIO
        if (ch == 1 || ch == 0)
          printf("ch = %d ltx = %d chi2min = %f chi2Limit = %f inverseSigmaSquared = %f\n",
                 ch,
                 ltx,
                 chi2min,
                 chi2Limit,
                 inverseSigmaSquared);
#endif

        // store into shared mem and run reduction
        // TODO: check if cooperative groups would be better
        // TODO: check if shuffling intrinsics are better
        shr_time_wgt[threadIdx.x] = inverseSigmaSquared;
        shr_time_max[threadIdx.x] = tmax * inverseSigmaSquared;
      } else {
        shr_time_wgt[threadIdx.x] = 0;
        shr_time_max[threadIdx.x] = 0;
      }
      __syncthreads();

      // reduce to compute time_max and time_wgt
      iter = nthreads_per_channel / 2 + nthreads_per_channel % 2;
      oddElements = nthreads_per_channel % 2;
      CMS_UNROLL_LOOP
      while (iter >= 1) {
        if (ltx < iter) {
          shr_time_wgt[threadIdx.x] = oddElements && (ltx == iter - 1 && ltx > 0)
                                          ? shr_time_wgt[threadIdx.x]
                                          : shr_time_wgt[threadIdx.x] + shr_time_wgt[threadIdx.x + iter];
          shr_time_max[threadIdx.x] = oddElements && (ltx == iter - 1 && ltx > 0)
                                          ? shr_time_max[threadIdx.x]
                                          : shr_time_max[threadIdx.x] + shr_time_max[threadIdx.x + iter];
          shrTimeMax[threadIdx.x] = oddElements && (ltx == iter - 1 && ltx > 0)
                                        ? shrTimeMax[threadIdx.x]
                                        : shrTimeMax[threadIdx.x] + shrTimeMax[threadIdx.x + iter];
          shrTimeWgt[threadIdx.x] = oddElements && (ltx == iter - 1 && ltx > 0)
                                        ? shrTimeWgt[threadIdx.x]
                                        : shrTimeWgt[threadIdx.x] + shrTimeWgt[threadIdx.x + iter];
        }

        __syncthreads();
        oddElements = iter % 2;
        iter = iter == 1 ? iter / 2 : iter / 2 + iter % 2;
      }

      // load from shared memory the 0th guy (will contain accumulated values)
      // compute
      // store into global mem
      if (ltx == 0) {
        const auto tmp_time_max = shr_time_max[threadIdx.x];
        const auto tmp_time_wgt = shr_time_wgt[threadIdx.x];

        // we are done if there number of time ratios is 0
        if (tmp_time_wgt == 0 && tmp_time_max == 0) {
          g_state[ch] = TimeComputationState::Finished;
          return;
        }

        // no div by 0
        const auto tMaxAlphaBeta = tmp_time_max / tmp_time_wgt;
        const auto tMaxErrorAlphaBeta = 1.0 / std::sqrt(tmp_time_wgt);

        tMaxAlphaBetas[ch] = tMaxAlphaBeta;
        tMaxErrorAlphaBetas[ch] = tMaxErrorAlphaBeta;
        g_accTimeMax[ch] = shrTimeMax[threadIdx.x];
        g_accTimeWgt[ch] = shrTimeWgt[threadIdx.x];
        g_state[ch] = TimeComputationState::NotFinished;

#ifdef DEBUG_TC_MAKERATIO
        printf("ch = %d time_max = %f time_wgt = %f\n", ch, tmp_time_max, tmp_time_wgt);
        printf("ch = %d tMaxAlphaBeta = %f tMaxErrorAlphaBeta = %f timeMax = %f timeWgt = %f\n",
               ch,
               tMaxAlphaBeta,
               tMaxErrorAlphaBeta,
               shrTimeMax[threadIdx.x],
               shrTimeWgt[threadIdx.x]);
#endif
      }
    }

    /// launch ctx parameters are
    /// 10 threads per channel, N channels per block, Y blocks
    /// TODO: do we need to keep the state around or can be removed?!
    //#define DEBUG_FINDAMPLCHI2_AND_FINISH
    __global__ void kernel_time_compute_findamplchi2_and_finish(
        SampleVector::Scalar const* sample_values,
        SampleVector::Scalar const* sample_value_errors,
        uint32_t const* dids_eb,
        uint32_t const* dids_ee,
        bool const* useless_samples,
        SampleVector::Scalar const* g_tMaxAlphaBeta,
        SampleVector::Scalar const* g_tMaxErrorAlphaBeta,
        SampleVector::Scalar const* g_accTimeMax,
        SampleVector::Scalar const* g_accTimeWgt,
        ConfigurationParameters::type const* amplitudeFitParametersEB,
        ConfigurationParameters::type const* amplitudeFitParametersEE,
        SampleVector::Scalar const* sumAAsNullHypot,
        SampleVector::Scalar const* sum0sNullHypot,
        SampleVector::Scalar const* chi2sNullHypot,
        TimeComputationState* g_state,
        SampleVector::Scalar* g_ampMaxAlphaBeta,
        SampleVector::Scalar* g_ampMaxError,
        SampleVector::Scalar* g_timeMax,
        SampleVector::Scalar* g_timeError,
        const int nchannels,
        uint32_t const offsetForInputs) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int gtx = threadIdx.x + blockIdx.x * blockDim.x;
      const int ch = gtx / nsamples;
      const int sample = threadIdx.x % nsamples;
      const auto* dids = ch >= offsetForInputs ? dids_ee : dids_eb;
      const int inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;

      // configure shared mem
      // per block, we need #threads per block * 2 * sizeof(ScalarType)
      // we run with N channels per block
      extern __shared__ char smem[];
      ScalarType* shr_sumAf = reinterpret_cast<ScalarType*>(smem);
      ScalarType* shr_sumff = shr_sumAf + blockDim.x;

      if (ch >= nchannels)
        return;

      auto state = g_state[ch];
      const auto did = DetId{dids[inputCh]};
      const auto* amplitudeFitParameters =
          did.subdetId() == EcalBarrel ? amplitudeFitParametersEB : amplitudeFitParametersEE;

      // TODO is that better than storing into global and launching another kernel
      // for the first 10 threads
      if (state == TimeComputationState::NotFinished) {
        const auto alpha = amplitudeFitParameters[0];
        const auto beta = amplitudeFitParameters[1];
        const auto alphabeta = alpha * beta;
        const auto invalphabeta = 1.0 / alphabeta;
        const auto tMaxAlphaBeta = g_tMaxAlphaBeta[ch];
        const auto sample_value = sample_values[gtx];
        const auto sample_value_error = sample_value_errors[gtx];
        const auto inverr2 =
            useless_samples[gtx] ? static_cast<ScalarType>(0) : 1.0 / (sample_value_error * sample_value_error);
        const auto offset = (static_cast<ScalarType>(sample) - tMaxAlphaBeta) * invalphabeta;
        const auto term1 = 1.0 + offset;
        const auto f = term1 > 1e-6 ? fast_expf(alpha * (fast_logf(term1) - offset)) : static_cast<ScalarType>(0.0);
        const auto sumAf = sample_value * (f * inverr2);
        const auto sumff = f * (f * inverr2);

        // store into shared mem
        shr_sumAf[threadIdx.x] = sumAf;
        shr_sumff[threadIdx.x] = sumff;
      } else {
        shr_sumAf[threadIdx.x] = 0;
        shr_sumff[threadIdx.x] = 0;
      }
      __syncthreads();

      // reduce
      // unroll completely here (but hardcoded)
      if (sample < 5) {
        shr_sumAf[threadIdx.x] += shr_sumAf[threadIdx.x + 5];
        shr_sumff[threadIdx.x] += shr_sumff[threadIdx.x + 5];
      }
      __syncthreads();

      if (sample < 2) {
        // will need to subtract for ltx = 3, we double count here
        shr_sumAf[threadIdx.x] += shr_sumAf[threadIdx.x + 2] + shr_sumAf[threadIdx.x + 3];
        shr_sumff[threadIdx.x] += shr_sumff[threadIdx.x + 2] + shr_sumff[threadIdx.x + 3];
      }
      __syncthreads();

      if (sample == 0) {
        // exit if the state is done
        // note, we do not exit before all __synchtreads are finished
        if (state == TimeComputationState::Finished) {
          g_timeMax[ch] = 5;
          g_timeError[ch] = -999;
          return;
        }

        // subtract to avoid double counting
        const auto sumff = shr_sumff[threadIdx.x] + shr_sumff[threadIdx.x + 1] - shr_sumff[threadIdx.x + 3];
        const auto sumAf = shr_sumAf[threadIdx.x] + shr_sumAf[threadIdx.x + 1] - shr_sumAf[threadIdx.x + 3];

        const auto ampMaxAlphaBeta = sumff > 0 ? sumAf / sumff : 0;
        const auto sumAA = sumAAsNullHypot[ch];
        const auto sum0 = sum0sNullHypot[ch];
        const auto nullChi2 = chi2sNullHypot[ch];
        if (sumff > 0) {
          const auto chi2AlphaBeta = (sumAA - sumAf * sumAf / sumff) / sum0;
          if (chi2AlphaBeta > nullChi2) {
            // null hypothesis is better
            state = TimeComputationState::Finished;
#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
            printf("ch = %d chi2AlphaBeta = %f nullChi2 = %f sumAA = %f sumAf = %f sumff = %f sum0 = %f\n",
                   ch,
                   chi2AlphaBeta,
                   nullChi2,
                   sumAA,
                   sumAf,
                   sumff,
                   sum0);
#endif
          }

          // store to global
          g_ampMaxAlphaBeta[ch] = ampMaxAlphaBeta;
        } else {
#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
          printf("ch = %d sum0 = %f sumAA = %f sumff = %f sumAf = %f\n", ch, sum0, sumAA, sumff, sumAf);
#endif
          state = TimeComputationState::Finished;
        }

        // store the state to global and finish calcs
        g_state[ch] = state;
        if (state == TimeComputationState::Finished) {
          // store default values into global
          g_timeMax[ch] = 5;
          g_timeError[ch] = -999;
#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
          printf("ch = %d finished state\n", ch);
#endif
          return;
        }

        const auto ampMaxError = g_ampMaxError[ch];
        const auto test_ratio = ampMaxAlphaBeta / ampMaxError;
        const auto accTimeMax = g_accTimeMax[ch];
        const auto accTimeWgt = g_accTimeWgt[ch];
        const auto tMaxAlphaBeta = g_tMaxAlphaBeta[ch];
        const auto tMaxErrorAlphaBeta = g_tMaxErrorAlphaBeta[ch];
        // branch to separate large vs small pulses
        // see cpu version for more info
        if (test_ratio > 5.0 && accTimeWgt > 0) {
          const auto tMaxRatio = accTimeWgt > 0 ? accTimeMax / accTimeWgt : static_cast<ScalarType>(0);
          const auto tMaxErrorRatio = accTimeWgt > 0 ? 1.0 / std::sqrt(accTimeWgt) : static_cast<ScalarType>(0);

          if (test_ratio > 10.0) {
            g_timeMax[ch] = tMaxRatio;
            g_timeError[ch] = tMaxErrorRatio;

#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
            printf("ch = %d tMaxRatio = %f tMaxErrorRatio = %f\n", ch, tMaxRatio, tMaxErrorRatio);
#endif
          } else {
            const auto timeMax = (tMaxAlphaBeta * (10.0 - ampMaxAlphaBeta / ampMaxError) +
                                  tMaxRatio * (ampMaxAlphaBeta / ampMaxError - 5.0)) /
                                 5.0;
            const auto timeError = (tMaxErrorAlphaBeta * (10.0 - ampMaxAlphaBeta / ampMaxError) +
                                    tMaxErrorRatio * (ampMaxAlphaBeta / ampMaxError - 5.0)) /
                                   5.0;
            state = TimeComputationState::Finished;
            g_state[ch] = state;
            g_timeMax[ch] = timeMax;
            g_timeError[ch] = timeError;

#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
            printf("ch = %d timeMax = %f timeError = %f\n", ch, timeMax, timeError);
#endif
          }
        } else {
          state = TimeComputationState::Finished;
          g_state[ch] = state;
          g_timeMax[ch] = tMaxAlphaBeta;
          g_timeError[ch] = tMaxErrorAlphaBeta;

#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
          printf("ch = %d tMaxAlphaBeta = %f tMaxErrorAlphaBeta = %f\n", ch, tMaxAlphaBeta, tMaxErrorAlphaBeta);
#endif
        }
      }
    }

    __global__ void kernel_time_compute_fixMGPAslew(uint16_t const* digis_eb,
                                                    uint16_t const* digis_ee,
                                                    SampleVector::Scalar* sample_values,
                                                    SampleVector::Scalar* sample_value_errors,
                                                    bool* useless_sample_values,
                                                    unsigned const int sample_mask,
                                                    const int nchannels,
                                                    uint32_t const offsetForInputs) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int gtx = threadIdx.x + blockIdx.x * blockDim.x;
      const int ch = gtx / nsamples;
      const int sample = threadIdx.x % nsamples;
      const int inputGtx = ch >= offsetForInputs ? gtx - offsetForInputs * nsamples : gtx;
      const auto* digis = ch >= offsetForInputs ? digis_ee : digis_eb;

      // remove thread for sample 0, oversubscribing is easier than ....
      if (ch >= nchannels || sample == 0)
        return;

      if (!use_sample(sample_mask, sample))
        return;

      const auto gainIdPrev = ecal::mgpa::gainId(digis[inputGtx - 1]);
      const auto gainIdNext = ecal::mgpa::gainId(digis[inputGtx]);
      if (gainIdPrev >= 1 && gainIdPrev <= 3 && gainIdNext >= 1 && gainIdNext <= 3 && gainIdPrev < gainIdNext) {
        sample_values[gtx - 1] = 0;
        sample_value_errors[gtx - 1] = 1e+9;
        useless_sample_values[gtx - 1] = true;
      }
    }

    __global__ void kernel_time_compute_ampl(SampleVector::Scalar const* sample_values,
                                             SampleVector::Scalar const* sample_value_errors,
                                             uint32_t const* dids,
                                             bool const* useless_samples,
                                             SampleVector::Scalar const* g_timeMax,
                                             SampleVector::Scalar const* amplitudeFitParametersEB,
                                             SampleVector::Scalar const* amplitudeFitParametersEE,
                                             SampleVector::Scalar* g_amplitudeMax,
                                             const int nchannels) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr ScalarType corr4 = 1.;
      constexpr ScalarType corr6 = 1.;
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int gtx = threadIdx.x + blockIdx.x * blockDim.x;
      const int ch = gtx / nsamples;
      const int sample = threadIdx.x % nsamples;

      if (ch >= nchannels)
        return;

      const auto did = DetId{dids[ch]};
      const auto* amplitudeFitParameters =
          did.subdetId() == EcalBarrel ? amplitudeFitParametersEB : amplitudeFitParametersEE;

      // configure shared mem
      extern __shared__ char smem[];
      ScalarType* shr_sum1 = reinterpret_cast<ScalarType*>(smem);
      auto* shr_sumA = shr_sum1 + blockDim.x;
      auto* shr_sumF = shr_sumA + blockDim.x;
      auto* shr_sumAF = shr_sumF + blockDim.x;
      auto* shr_sumFF = shr_sumAF + blockDim.x;

      const auto alpha = amplitudeFitParameters[0];
      const auto beta = amplitudeFitParameters[1];
      const auto timeMax = g_timeMax[ch];
      const auto pedestalLimit = timeMax - (alpha * beta) - 1.0;
      const auto sample_value = sample_values[gtx];
      const auto sample_value_error = sample_value_errors[gtx];
      const auto inverr2 =
          sample_value_error > 0 ? 1. / (sample_value_error * sample_value_error) : static_cast<ScalarType>(0);
      const auto termOne = 1 + (sample - timeMax) / (alpha * beta);
      const auto f = termOne > 1.e-5 ? fast_expf(alpha * fast_logf(termOne) - (sample - timeMax) / beta)
                                     : static_cast<ScalarType>(0.);

      bool const cond = ((sample < pedestalLimit) || (f > 0.6 * corr6 && sample <= timeMax) ||
                         (f > 0.4 * corr4 && sample >= timeMax)) &&
                        !useless_samples[gtx];

      // store into shared mem
      shr_sum1[threadIdx.x] = cond ? inverr2 : static_cast<ScalarType>(0);
      shr_sumA[threadIdx.x] = cond ? sample_value * inverr2 : static_cast<ScalarType>(0);
      shr_sumF[threadIdx.x] = cond ? f * inverr2 : static_cast<ScalarType>(0);
      shr_sumAF[threadIdx.x] = cond ? (f * inverr2) * sample_value : static_cast<ScalarType>(0);
      shr_sumFF[threadIdx.x] = cond ? f * (f * inverr2) : static_cast<ScalarType>(0);

      // reduction
      if (sample <= 4) {
        shr_sum1[threadIdx.x] += shr_sum1[threadIdx.x + 5];
        shr_sumA[threadIdx.x] += shr_sumA[threadIdx.x + 5];
        shr_sumF[threadIdx.x] += shr_sumF[threadIdx.x + 5];
        shr_sumAF[threadIdx.x] += shr_sumAF[threadIdx.x + 5];
        shr_sumFF[threadIdx.x] += shr_sumFF[threadIdx.x + 5];
      }
      __syncthreads();

      if (sample < 2) {
        // note: we double count sample 3
        shr_sum1[threadIdx.x] += shr_sum1[threadIdx.x + 2] + shr_sum1[threadIdx.x + 3];
        shr_sumA[threadIdx.x] += shr_sumA[threadIdx.x + 2] + shr_sumA[threadIdx.x + 3];
        shr_sumF[threadIdx.x] += shr_sumF[threadIdx.x + 2] + shr_sumF[threadIdx.x + 3];
        shr_sumAF[threadIdx.x] += shr_sumAF[threadIdx.x + 2] + shr_sumAF[threadIdx.x + 3];
        shr_sumFF[threadIdx.x] += shr_sumFF[threadIdx.x + 2] + shr_sumFF[threadIdx.x + 3];
      }
      __syncthreads();

      if (sample == 0) {
        const auto sum1 = shr_sum1[threadIdx.x] + shr_sum1[threadIdx.x + 1] - shr_sum1[threadIdx.x + 3];
        const auto sumA = shr_sumA[threadIdx.x] + shr_sumA[threadIdx.x + 1] - shr_sumA[threadIdx.x + 3];
        const auto sumF = shr_sumF[threadIdx.x] + shr_sumF[threadIdx.x + 1] - shr_sumF[threadIdx.x + 3];
        const auto sumAF = shr_sumAF[threadIdx.x] + shr_sumAF[threadIdx.x + 1] - shr_sumAF[threadIdx.x + 3];
        const auto sumFF = shr_sumFF[threadIdx.x] + shr_sumFF[threadIdx.x + 1] - shr_sumFF[threadIdx.x + 3];

        const auto denom = sumFF * sum1 - sumF * sumF;
        const auto condForDenom = sum1 > 0 && std::abs(denom) > 1.e-20;
        const auto amplitudeMax = condForDenom ? (sumAF * sum1 - sumA * sumF) / denom : static_cast<ScalarType>(0.);

        // store into global mem
        g_amplitudeMax[ch] = amplitudeMax;
      }
    }

    //#define ECAL_RECO_CUDA_TC_INIT_DEBUG
    __global__ void kernel_time_computation_init(uint16_t const* digis_eb,
                                                 uint32_t const* dids_eb,
                                                 uint16_t const* digis_ee,
                                                 uint32_t const* dids_ee,
                                                 float const* rms_x12,
                                                 float const* rms_x6,
                                                 float const* rms_x1,
                                                 float const* mean_x12,
                                                 float const* mean_x6,
                                                 float const* mean_x1,
                                                 float const* gain12Over6,
                                                 float const* gain6Over1,
                                                 SampleVector::Scalar* sample_values,
                                                 SampleVector::Scalar* sample_value_errors,
                                                 SampleVector::Scalar* ampMaxError,
                                                 bool* useless_sample_values,
                                                 char* pedestal_nums,
                                                 uint32_t const offsetForHashes,
                                                 uint32_t const offsetForInputs,
                                                 unsigned const int sample_maskEB,
                                                 unsigned const int sample_maskEE,
                                                 int nchannels) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int tx = threadIdx.x + blockDim.x * blockIdx.x;
      const int ch = tx / nsamples;
      const int inputTx = ch >= offsetForInputs ? tx - offsetForInputs * nsamples : tx;
      const int inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;
      const auto* digis = ch >= offsetForInputs ? digis_ee : digis_eb;
      const auto* dids = ch >= offsetForInputs ? dids_ee : dids_eb;

      // threads that return here should not affect the __syncthreads() below since they have exitted the kernel
      if (ch >= nchannels)
        return;

      // indices/inits
      const int sample = tx % nsamples;
      const int input_ch_start = inputCh * nsamples;
      SampleVector::Scalar pedestal = 0.;
      int num = 0;

      // configure shared mem
      extern __shared__ char smem[];
      ScalarType* shrSampleValues = reinterpret_cast<SampleVector::Scalar*>(smem);
      ScalarType* shrSampleValueErrors = shrSampleValues + blockDim.x;

      // 0 and 1 sample values
      const auto adc0 = ecal::mgpa::adc(digis[input_ch_start]);
      const auto gainId0 = ecal::mgpa::gainId(digis[input_ch_start]);
      const auto adc1 = ecal::mgpa::adc(digis[input_ch_start + 1]);
      const auto gainId1 = ecal::mgpa::gainId(digis[input_ch_start + 1]);
      const auto did = DetId{dids[inputCh]};
      const auto isBarrel = did.subdetId() == EcalBarrel;
      const auto sample_mask = did.subdetId() == EcalBarrel ? sample_maskEB : sample_maskEE;
      const auto hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                     : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

      // set pedestal
      // TODO this branch is non-divergent for a group of 10 threads
      if (gainId0 == 1 && use_sample(sample_mask, 0)) {
        pedestal = static_cast<SampleVector::Scalar>(adc0);
        num = 1;

        const auto diff = adc1 - adc0;
        if (gainId1 == 1 && use_sample(sample_mask, 1) && std::abs(diff) < 3 * rms_x12[hashedId]) {
          pedestal = (pedestal + static_cast<SampleVector::Scalar>(adc1)) / 2.0;
          num = 2;
        }
      } else {
        pedestal = mean_x12[ch];
      }

      // ped subtracted and gain-renormalized samples.
      const auto gainId = ecal::mgpa::gainId(digis[inputTx]);
      const auto adc = ecal::mgpa::adc(digis[inputTx]);

      bool bad = false;
      SampleVector::Scalar sample_value, sample_value_error;
      // TODO divergent branch
      // TODO: piece below is general both for amplitudes and timing
      // potentially there is a way to reduce the amount of code...
      if (!use_sample(sample_mask, sample)) {
        bad = true;
        sample_value = 0;
        sample_value_error = 0;
      } else if (gainId == 1) {
        sample_value = static_cast<SampleVector::Scalar>(adc) - pedestal;
        sample_value_error = rms_x12[hashedId];
      } else if (gainId == 2) {
        sample_value = (static_cast<SampleVector::Scalar>(adc) - mean_x6[hashedId]) * gain12Over6[hashedId];
        sample_value_error = rms_x6[hashedId] * gain12Over6[hashedId];
      } else if (gainId == 3) {
        sample_value =
            (static_cast<SampleVector::Scalar>(adc) - mean_x1[hashedId]) * gain6Over1[hashedId] * gain12Over6[hashedId];
        sample_value_error = rms_x1[hashedId] * gain6Over1[hashedId] * gain12Over6[hashedId];
      } else {
        sample_value = 0;
        sample_value_error = 0;
        bad = true;
      }

      // TODO: make sure we save things correctly when sample is useless
      const auto useless_sample = (sample_value_error <= 0) | bad;
      useless_sample_values[tx] = useless_sample;
      sample_values[tx] = sample_value;
      sample_value_errors[tx] = useless_sample ? 1e+9 : sample_value_error;

      // DEBUG
#ifdef ECAL_RECO_CUDA_TC_INIT_DEBUG
      if (ch == 0) {
        printf("sample = %d sample_value = %f sample_value_error = %f useless = %c\n",
               sample,
               sample_value,
               sample_value_error,
               useless_sample ? '1' : '0');
      }
#endif

      // store into the shared mem
      shrSampleValues[threadIdx.x] = sample_value_error > 0 ? sample_value : std::numeric_limits<ScalarType>::min();
      shrSampleValueErrors[threadIdx.x] = sample_value_error;
      __syncthreads();

      // perform the reduction with min
      if (sample < 5) {
        // note, if equal -> we keep the value with lower sample as for cpu
        shrSampleValueErrors[threadIdx.x] = shrSampleValues[threadIdx.x] < shrSampleValues[threadIdx.x + 5]
                                                ? shrSampleValueErrors[threadIdx.x + 5]
                                                : shrSampleValueErrors[threadIdx.x];
        shrSampleValues[threadIdx.x] = std::max(shrSampleValues[threadIdx.x], shrSampleValues[threadIdx.x + 5]);
      }
      __syncthreads();

      // a bit of an overkill, but easier than to compare across 3 values
      if (sample < 3) {
        shrSampleValueErrors[threadIdx.x] = shrSampleValues[threadIdx.x] < shrSampleValues[threadIdx.x + 3]
                                                ? shrSampleValueErrors[threadIdx.x + 3]
                                                : shrSampleValueErrors[threadIdx.x];
        shrSampleValues[threadIdx.x] = std::max(shrSampleValues[threadIdx.x], shrSampleValues[threadIdx.x + 3]);
      }
      __syncthreads();

      if (sample < 2) {
        shrSampleValueErrors[threadIdx.x] = shrSampleValues[threadIdx.x] < shrSampleValues[threadIdx.x + 2]
                                                ? shrSampleValueErrors[threadIdx.x + 2]
                                                : shrSampleValueErrors[threadIdx.x];
        shrSampleValues[threadIdx.x] = std::max(shrSampleValues[threadIdx.x], shrSampleValues[threadIdx.x + 2]);
      }
      __syncthreads();

      if (sample == 0) {
        // we only needd the max error
        const auto maxSampleValueError = shrSampleValues[threadIdx.x] < shrSampleValues[threadIdx.x + 1]
                                             ? shrSampleValueErrors[threadIdx.x + 1]
                                             : shrSampleValueErrors[threadIdx.x];

        // # pedestal samples used
        pedestal_nums[ch] = num;
        // this is used downstream
        ampMaxError[ch] = maxSampleValueError;

        // DEBUG
#ifdef ECAL_RECO_CUDA_TC_INIT_DEBUG
        if (ch == 0) {
          printf("pedestal_nums = %d ampMaxError = %f\n", num, maxSampleValueError);
        }
#endif
      }
    }

    ///
    /// launch context parameters: 1 thread per channel
    ///
    //#define DEBUG_TIME_CORRECTION
    __global__ void kernel_time_correction_and_finalize(
        //        SampleVector::Scalar const* g_amplitude,
        ::ecal::reco::StorageScalarType const* g_amplitudeEB,
        ::ecal::reco::StorageScalarType const* g_amplitudeEE,
        uint16_t const* digis_eb,
        uint32_t const* dids_eb,
        uint16_t const* digis_ee,
        uint32_t const* dids_ee,
        float const* amplitudeBinsEB,
        float const* amplitudeBinsEE,
        float const* shiftBinsEB,
        float const* shiftBinsEE,
        SampleVector::Scalar const* g_timeMax,
        SampleVector::Scalar const* g_timeError,
        float const* g_rms_x12,
        float const* timeCalibConstant,
        float* g_jitterEB,
        float* g_jitterEE,
        float* g_jitterErrorEB,
        float* g_jitterErrorEE,
        uint32_t* flagsEB,
        uint32_t* flagsEE,
        const int amplitudeBinsSizeEB,
        const int amplitudeBinsSizeEE,
        ConfigurationParameters::type const timeConstantTermEB,
        ConfigurationParameters::type const timeConstantTermEE,
        float const offsetTimeValueEB,
        float const offsetTimeValueEE,
        ConfigurationParameters::type const timeNconstEB,
        ConfigurationParameters::type const timeNconstEE,
        ConfigurationParameters::type const amplitudeThresholdEB,
        ConfigurationParameters::type const amplitudeThresholdEE,
        ConfigurationParameters::type const outOfTimeThreshG12pEB,
        ConfigurationParameters::type const outOfTimeThreshG12pEE,
        ConfigurationParameters::type const outOfTimeThreshG12mEB,
        ConfigurationParameters::type const outOfTimeThreshG12mEE,
        ConfigurationParameters::type const outOfTimeThreshG61pEB,
        ConfigurationParameters::type const outOfTimeThreshG61pEE,
        ConfigurationParameters::type const outOfTimeThreshG61mEB,
        ConfigurationParameters::type const outOfTimeThreshG61mEE,
        uint32_t const offsetForHashes,
        uint32_t const offsetForInputs,
        const int nchannels) {
      using ScalarType = SampleVector::Scalar;

      // constants
      constexpr int nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      const int gtx = threadIdx.x + blockIdx.x * blockDim.x;
      const int inputGtx = gtx >= offsetForInputs ? gtx - offsetForInputs : gtx;
      const auto* dids = gtx >= offsetForInputs ? dids_ee : dids_eb;
      const auto& digis = gtx >= offsetForInputs ? digis_ee : digis_eb;

      // filter out outside of range threads
      if (gtx >= nchannels)
        return;

// need to ref the right ptrs
#define ARRANGE(var) auto* var = gtx >= offsetForInputs ? var##EE : var##EB
      ARRANGE(g_amplitude);
      ARRANGE(g_jitter);
      ARRANGE(g_jitterError);
      ARRANGE(flags);
#undef ARRANGE

      const auto did = DetId{dids[inputGtx]};
      const auto isBarrel = did.subdetId() == EcalBarrel;
      const auto hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                     : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());
      const auto* amplitudeBins = isBarrel ? amplitudeBinsEB : amplitudeBinsEE;
      const auto* shiftBins = isBarrel ? shiftBinsEB : shiftBinsEE;
      const auto amplitudeBinsSize = isBarrel ? amplitudeBinsSizeEB : amplitudeBinsSizeEE;
      const auto timeConstantTerm = isBarrel ? timeConstantTermEB : timeConstantTermEE;
      const auto timeNconst = isBarrel ? timeNconstEB : timeNconstEE;
      const auto offsetTimeValue = isBarrel ? offsetTimeValueEB : offsetTimeValueEE;
      const auto amplitudeThreshold = isBarrel ? amplitudeThresholdEB : amplitudeThresholdEE;
      const auto outOfTimeThreshG12p = isBarrel ? outOfTimeThreshG12pEB : outOfTimeThreshG12pEE;
      const auto outOfTimeThreshG12m = isBarrel ? outOfTimeThreshG12mEB : outOfTimeThreshG12mEE;
      const auto outOfTimeThreshG61p = isBarrel ? outOfTimeThreshG61pEB : outOfTimeThreshG61pEE;
      const auto outOfTimeThreshG61m = isBarrel ? outOfTimeThreshG61mEB : outOfTimeThreshG61mEE;

      // load some
      const auto amplitude = g_amplitude[inputGtx];
      const auto rms_x12 = g_rms_x12[hashedId];
      const auto timeCalibConst = timeCalibConstant[hashedId];

      int myBin = -1;
      for (int bin = 0; bin < amplitudeBinsSize; bin++) {
        if (amplitude > amplitudeBins[bin])
          myBin = bin;
        else
          break;
      }

      ScalarType correction = 0;
      if (myBin == -1) {
        correction = shiftBins[0];
      } else if (myBin == amplitudeBinsSize - 1) {
        correction = shiftBins[myBin];
      } else {
        correction = shiftBins[myBin + 1] - shiftBins[myBin];
        correction *= (amplitude - amplitudeBins[myBin]) / (amplitudeBins[myBin + 1] - amplitudeBins[myBin]);
        correction += shiftBins[myBin];
      }

      // correction * 1./25.
      correction = correction * 0.04;
      const auto timeMax = g_timeMax[gtx];
      const auto timeError = g_timeError[gtx];
      const auto jitter = timeMax - 5 + correction;
      const auto jitterError =
          std::sqrt(timeError * timeError + timeConstantTerm * timeConstantTerm * 0.04 * 0.04);  // 0.04 = 1./25.

#ifdef DEBUG_TIME_CORRECTION
      printf("ch = %d timeMax = %f timeError = %f jitter = %f correction = %f\n",
             gtx,
             timeMax,
             timeError,
             jitter,
             correction);
//    }
#endif

      // store back to  global
      g_jitter[inputGtx] = jitter;
      g_jitterError[inputGtx] = jitterError;

      // set the flag
      // TODO: replace with something more efficient (if required),
      // for now just to make it work
      if (amplitude > amplitudeThreshold * rms_x12) {
        auto threshP = outOfTimeThreshG12p;
        auto threshM = outOfTimeThreshG12m;
        if (amplitude > 3000.) {
          for (int isample = 0; isample < nsamples; isample++) {
            int gainid = ecal::mgpa::gainId(digis[nsamples * inputGtx + isample]);
            if (gainid != 1) {
              threshP = outOfTimeThreshG61p;
              threshM = outOfTimeThreshG61m;
              break;
            }
          }
        }

        const auto correctedTime = (timeMax - 5) * 25 + timeCalibConst + offsetTimeValue;
        const auto nterm = timeNconst * rms_x12 / amplitude;
        const auto sigmat = std::sqrt(nterm * nterm + timeConstantTerm * timeConstantTerm);
        if (correctedTime > sigmat * threshP || correctedTime < -sigmat * threshM)
          flags[inputGtx] |= 0x1 << EcalUncalibratedRecHit::kOutOfTime;
      }
    }

  }  // namespace multifit
}  // namespace ecal
