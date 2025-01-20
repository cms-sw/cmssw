#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_TimeComputationKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_TimeComputationKernels_h

#include <cassert>
#include <cmath>
#include <limits>

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EigenMatrixTypes_gpu.h"

#include "DeclsForKernels.h"
#include "KernelHelpers.h"
#include "EcalMultifitParameters.h"

//#define ECAL_RECO_ALPAKA_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool use_sample(unsigned int sample_mask, unsigned int sample) {
    return sample_mask & (0x1 << (EcalDataFrame::MAXSAMPLES - (sample + 1)));
  }

  ALPAKA_FN_ACC constexpr float fast_expf(float x) { return unsafe_expf<6>(x); }
  ALPAKA_FN_ACC constexpr float fast_logf(float x) { return unsafe_logf<7>(x); }

  class Kernel_time_compute_nullhypot {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ScalarType* const sample_values,
                                  ScalarType* const sample_value_errors,
                                  bool* const useless_sample_values,
                                  ScalarType* chi2s,
                                  ScalarType* sum0s,
                                  ScalarType* sumAAs,
                                  uint32_t const nchannels) const {
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

      // shared mem inits
      auto* s_sum0 = alpaka::getDynSharedMem<char>(acc);
      auto* s_sum1 = reinterpret_cast<ScalarType*>(s_sum0 + elemsPerBlock);
      auto* s_sumA = s_sum1 + elemsPerBlock;
      auto* s_sumAA = s_sumA + elemsPerBlock;

      for (auto txforward : cms::alpakatools::uniform_elements(acc, nchannels * nsamples)) {
        // go backwards through the loop to have valid values for shared variables when reading from higher element indices in serial execution
        auto tx = nchannels * nsamples - 1 - txforward;
        auto const ch = tx / nsamples;

        auto const sample = tx % nsamples;
        auto const ltx = tx % elemsPerBlock;

        // TODO make sure no div by 0
        auto const inv_error =
            useless_sample_values[tx] ? 0. : 1. / (sample_value_errors[tx] * sample_value_errors[tx]);
        auto const sample_value = sample_values[tx];
        s_sum0[ltx] = useless_sample_values[tx] ? 0 : 1;
        s_sum1[ltx] = inv_error;
        s_sumA[ltx] = sample_value * inv_error;
        s_sumAA[ltx] = sample_value * sample_value * inv_error;
        alpaka::syncBlockThreads(acc);

        // 5 threads for [0, 4] samples
        if (sample < 5) {
          s_sum0[ltx] += s_sum0[ltx + 5];
          s_sum1[ltx] += s_sum1[ltx + 5];
          s_sumA[ltx] += s_sumA[ltx + 5];
          s_sumAA[ltx] += s_sumAA[ltx + 5];
        }
        alpaka::syncBlockThreads(acc);

        if (sample < 2) {
          // note double counting of sample 3
          s_sum0[ltx] += s_sum0[ltx + 2] + s_sum0[ltx + 3];
          s_sum1[ltx] += s_sum1[ltx + 2] + s_sum1[ltx + 3];
          s_sumA[ltx] += s_sumA[ltx + 2] + s_sumA[ltx + 3];
          s_sumAA[ltx] += s_sumAA[ltx + 2] + s_sumAA[ltx + 3];
        }
        alpaka::syncBlockThreads(acc);

        if (sample == 0) {
          // note, subtract to remove the double counting of sample == 3
          auto const sum0 = s_sum0[ltx] + s_sum0[ltx + 1] - s_sum0[ltx + 3];
          auto const sum1 = s_sum1[ltx] + s_sum1[ltx + 1] - s_sum1[ltx + 3];
          auto const sumA = s_sumA[ltx] + s_sumA[ltx + 1] - s_sumA[ltx + 3];
          auto const sumAA = s_sumAA[ltx] + s_sumAA[ltx + 1] - s_sumAA[ltx + 3];
          auto const chi2 = sum0 > 0 ? (sumAA - sumA * sumA / sum1) / sum0 : static_cast<ScalarType>(0);
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
    }
  };

  //
  // launch ctx parameters are
  // 45 threads per channel, X channels per block, Y blocks
  // 45 comes from: 10 samples for i <- 0 to 9 and for j <- i+1 to 9
  // TODO: it might be much beter to use 32 threads per channel instead of 45
  // to simplify the synchronization
  class Kernel_time_compute_makeratio {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  ScalarType* const sample_values,
                                  ScalarType* const sample_value_errors,
                                  bool* const useless_sample_values,
                                  char* const pedestal_nums,
                                  ScalarType* const sumAAsNullHypot,
                                  ScalarType* const sum0sNullHypot,
                                  ScalarType* tMaxAlphaBetas,
                                  ScalarType* tMaxErrorAlphaBetas,
                                  ScalarType* g_accTimeMax,
                                  ScalarType* g_accTimeWgt,
                                  TimeComputationState* g_state,
                                  EcalMultifitParameters const* paramsDev,
                                  ConfigurationParameters::type const timeFitLimits_firstEB,
                                  ConfigurationParameters::type const timeFitLimits_firstEE,
                                  ConfigurationParameters::type const timeFitLimits_secondEB,
                                  ConfigurationParameters::type const timeFitLimits_secondEE) const {
      // constants
      constexpr uint32_t nchannels_per_block = 10;
      constexpr auto nthreads_per_channel = nchannels_per_block * (nchannels_per_block - 1) / 2;
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;
      auto const nchannels = digisDevEB.size() + digisDevEE.size();
      auto const offsetForInputs = digisDevEB.size();
      auto const totalElements = nthreads_per_channel * nchannels;

      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      ALPAKA_ASSERT_ACC(nthreads_per_channel * nchannels_per_block == elemsPerBlock);

      auto* shr_chi2s = alpaka::getDynSharedMem<ScalarType>(acc);
      auto* shr_time_wgt = shr_chi2s + elemsPerBlock;
      auto* shr_time_max = shr_time_wgt + elemsPerBlock;
      auto* shrTimeMax = shr_time_max + elemsPerBlock;
      auto* shrTimeWgt = shrTimeMax + elemsPerBlock;
      auto* shr_chi2 = shrTimeWgt + elemsPerBlock;
      auto* shr_tmax = shr_chi2 + elemsPerBlock;
      auto* shr_tmaxerr = shr_tmax + elemsPerBlock;
      auto* shr_condForUselessSamples = reinterpret_cast<bool*>(shr_tmaxerr + elemsPerBlock);
      auto* shr_internalCondForSkipping1 = shr_condForUselessSamples + elemsPerBlock;
      auto* shr_internalCondForSkipping2 = shr_internalCondForSkipping1 + elemsPerBlock;

      for (auto block : cms::alpakatools::uniform_groups(acc, totalElements)) {
        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const ch = idx.global / nthreads_per_channel;
          auto const ltx = idx.global % nthreads_per_channel;

          auto const ch_start = ch * nsamples;
          auto const inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;
          auto const* dids = ch >= offsetForInputs ? digisDevEE.id() : digisDevEB.id();

          auto const did = DetId{dids[inputCh]};
          auto const isBarrel = did.subdetId() == EcalBarrel;
          auto* const amplitudeFitParameters =
              isBarrel ? paramsDev->amplitudeFitParamsEB.data() : paramsDev->amplitudeFitParamsEE.data();
          auto* const timeFitParameters =
              isBarrel ? paramsDev->timeFitParamsEB.data() : paramsDev->timeFitParamsEE.data();
          auto const timeFitParameters_size =
              isBarrel ? paramsDev->timeFitParamsEB.size() : paramsDev->timeFitParamsEE.size();
          auto const timeFitLimits_first = isBarrel ? timeFitLimits_firstEB : timeFitLimits_firstEE;
          auto const timeFitLimits_second = isBarrel ? timeFitLimits_secondEB : timeFitLimits_secondEE;

          // map tx -> (sample_i, sample_j)
          int sample_i = 0;
          int sample_j = 0;
          if (ltx <= 8) {
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
          } else {
            // FIXME this needs a more portable solution, that wraps abort() / __trap() / throw depending on the back-end
            ALPAKA_ASSERT_ACC(false);
          }

          auto const tx_i = ch_start + sample_i;
          auto const tx_j = ch_start + sample_j;

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
          shrTimeMax[idx.local] = 0;
          shrTimeWgt[idx.local] = 0;

          bool internalCondForSkipping1 = true;
          bool internalCondForSkipping2 = true;
          if (!condForUselessSamples) {
            auto const rtmp = sample_values[tx_i] / sample_values[tx_j];
            auto const invampl_i = 1. / sample_values[tx_i];
            auto const relErr2_i = sample_value_errors[tx_i] * sample_value_errors[tx_i] * invampl_i * invampl_i;
            auto const invampl_j = 1. / sample_values[tx_j];
            auto const relErr2_j = sample_value_errors[tx_j] * sample_value_errors[tx_j] * invampl_j * invampl_j;
            auto const err1 = rtmp * rtmp * (relErr2_i + relErr2_j);
            auto err2 =
                sample_value_errors[tx_j] * (sample_values[tx_i] - sample_values[tx_j]) * (invampl_j * invampl_j);
            // TODO non-divergent branch for a block if each block has 1 channel
            // otherwise non-divergent for groups of 45 threads
            // at this point, pedestal_nums[ch] can be either 0, 1 or 2
            if (pedestal_nums[ch] == 2)
              err2 *= err2 * 0.5;
            auto const err3 = (0.289 * 0.289) * (invampl_j * invampl_j);
            auto const total_error = std::sqrt(err1 + err2 + err3);

            auto const alpha = amplitudeFitParameters[0];
            auto const beta = amplitudeFitParameters[1];
            auto const alphabeta = alpha * beta;
            auto const invalphabeta = 1. / alphabeta;

            // variables instead of a struct
            auto const ratio_index = sample_i;
            auto const ratio_step = sample_j - sample_i;
            auto const ratio_value = rtmp;
            auto const ratio_error = total_error;

            auto const rlim_i_j = fast_expf(static_cast<ScalarType>(sample_j - sample_i) / beta) - 0.001;
            internalCondForSkipping1 = !(total_error < 1. && rtmp > 0.001 && rtmp < rlim_i_j);
            if (!internalCondForSkipping1) {
              //
              // precompute.
              // in cpu version this was done conditionally
              // however easier to do it here (precompute) and then just filter out
              // if not needed
              //
              auto const l_timeFitLimits_first = timeFitLimits_first;
              auto const l_timeFitLimits_second = timeFitLimits_second;
              if (ratio_step == 1 && ratio_value >= l_timeFitLimits_first && ratio_value <= l_timeFitLimits_second) {
                auto const time_max_i = static_cast<ScalarType>(ratio_index);
                auto u = timeFitParameters[timeFitParameters_size - 1];
                CMS_UNROLL_LOOP
                for (int k = timeFitParameters_size - 2; k >= 0; --k)
                  u = u * ratio_value + timeFitParameters[k];

                auto du = (timeFitParameters_size - 1) * (timeFitParameters[timeFitParameters_size - 1]);
                for (int k = timeFitParameters_size - 2; k >= 1; --k)
                  du = du * ratio_value + k * timeFitParameters[k];

                auto const error2 = ratio_error * ratio_error * du * du;
                auto const time_max = error2 > 0 ? (time_max_i - u) / error2 : static_cast<ScalarType>(0);
                auto const time_wgt = error2 > 0 ? 1. / error2 : static_cast<ScalarType>(0);

                // store into shared mem
                // note, this name is essentially identical to the one used
                // below.
                shrTimeMax[idx.local] = error2 > 0 ? time_max : 0;
                shrTimeWgt[idx.local] = error2 > 0 ? time_wgt : 0;
              } else {
                shrTimeMax[idx.local] = 0;
                shrTimeWgt[idx.local] = 0;
              }

              // continue with ratios
              auto const stepOverBeta = static_cast<ScalarType>(ratio_step) / beta;
              auto const offset = static_cast<ScalarType>(ratio_index) + alphabeta;
              auto const rmin = std::max(ratio_value - ratio_error, 0.001);
              auto const rmax =
                  std::min(ratio_value + ratio_error, fast_expf(static_cast<ScalarType>(ratio_step) / beta) - 0.001);
              auto const time1 = offset - ratio_step / (fast_expf((stepOverBeta - fast_logf(rmin)) / alpha) - 1.);
              auto const time2 = offset - ratio_step / (fast_expf((stepOverBeta - fast_logf(rmax)) / alpha) - 1.);

              // set these guys
              tmax = 0.5 * (time1 + time2);
              tmaxerr = 0.5 * std::sqrt((time1 - time2) * (time1 - time2));
#ifdef DEBUG_TC_MAKERATIO
              if (ch == 1 || ch == 0)
                printf(
                    "ch = %d ltx = %d tmax = %f tmaxerr = %f time1 = %f time2 = %f offset = %f rmin = %f rmax = "
                    "%f\n",
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

              ScalarType sumAf = 0;
              ScalarType sumff = 0;
              const int itmin = std::max(-1, static_cast<int>(std::floor(tmax - alphabeta)));
              auto loffset = (static_cast<ScalarType>(itmin) - tmax) * invalphabeta;
              // TODO: data dependence
              for (int it = itmin + 1; it < nsamples; ++it) {
                loffset += invalphabeta;
                if (useless_sample_values[ch_start + it])
                  continue;
                auto const inverr2 = 1. / (sample_value_errors[ch_start + it] * sample_value_errors[ch_start + it]);
                auto const term1 = 1. + loffset;
                auto const f = (term1 > 1e-6) ? fast_expf(alpha * (fast_logf(term1) - loffset)) : 0;
                sumAf += sample_values[ch_start + it] * (f * inverr2);
                sumff += f * (f * inverr2);
              }

              auto const sumAA = sumAAsNullHypot[ch];
              auto const sum0 = sum0sNullHypot[ch];
              chi2 = sumAA;
              // TODO: sum0 can not be 0 below, need to introduce the check upfront
              if (sumff > 0) {
                chi2 = sumAA - sumAf * (sumAf / sumff);
              }
              chi2 /= sum0;

#ifdef DEBUG_TC_MAKERATIO
              if (ch == 1 || ch == 0)
                printf(
                    "ch = %d ltx = %d sumAf = %f sumff = %f sumAA = %f sum0 = %d tmax = %f tmaxerr = %f chi2 = "
                    "%f\n",
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
          shr_chi2s[idx.local] = chi2;
          shr_chi2[idx.local] = chi2;
          shr_tmax[idx.local] = tmax;
          shr_tmaxerr[idx.local] = tmaxerr;
          shr_condForUselessSamples[idx.local] = condForUselessSamples;
          shr_internalCondForSkipping1[idx.local] = internalCondForSkipping1;
          shr_internalCondForSkipping2[idx.local] = internalCondForSkipping2;
        }

        alpaka::syncBlockThreads(acc);

        // find min chi2 - quite crude for now
        // TODO validate/check
        auto iter = nthreads_per_channel / 2 + nthreads_per_channel % 2;
        bool oddElements = nthreads_per_channel % 2;
        CMS_UNROLL_LOOP
        while (iter >= 1) {
          for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
            auto const ltx = idx.global % nthreads_per_channel;

            if (ltx < iter && !(oddElements && (ltx == iter - 1 && ltx > 0))) {
              // for odd ns, the last guy will just store itself
              // exception is for ltx == 0 and iter==1
              shr_chi2s[idx.local] = std::min(shr_chi2s[idx.local], shr_chi2s[idx.local + iter]);
            }
          }
          alpaka::syncBlockThreads(acc);

          oddElements = iter % 2;
          iter = iter == 1 ? iter / 2 : iter / 2 + iter % 2;
        }

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const ltx = idx.global % nthreads_per_channel;

          // get precomputedflags for this element from shared memory
          auto const condForUselessSamples = shr_condForUselessSamples[idx.local];
          auto const internalCondForSkipping1 = shr_internalCondForSkipping1[idx.local];
          auto const internalCondForSkipping2 = shr_internalCondForSkipping2[idx.local];
          // filter out inactive or useless samples threads
          if (!condForUselessSamples && !internalCondForSkipping1 && !internalCondForSkipping2) {
            // min chi2, now compute weighted average of tmax measurements
            // see cpu version for more explanation
            auto const chi2 = shr_chi2[idx.local];
            auto const chi2min = shr_chi2s[idx.local - ltx];
            auto const chi2Limit = chi2min + 1.;
            auto const tmaxerr = shr_tmaxerr[idx.local];
            auto const inverseSigmaSquared = chi2 < chi2Limit ? 1. / (tmaxerr * tmaxerr) : 0.;

#ifdef DEBUG_TC_MAKERATIO
            if (ch == 1 || ch == 0) {
              auto const ch = idx.global / nthreads_per_channel;
              printf("ch = %d ltx = %d chi2min = %f chi2Limit = %f inverseSigmaSquared = %f\n",
                     ch,
                     ltx,
                     chi2min,
                     chi2Limit,
                     inverseSigmaSquared);
            }
#endif

            // store into shared mem and run reduction
            // TODO: check if cooperative groups would be better
            // TODO: check if shuffling intrinsics are better
            auto const tmax = shr_tmax[idx.local];
            shr_time_wgt[idx.local] = inverseSigmaSquared;
            shr_time_max[idx.local] = tmax * inverseSigmaSquared;
          } else {
            shr_time_wgt[idx.local] = 0;
            shr_time_max[idx.local] = 0;
          }
        }

        alpaka::syncBlockThreads(acc);

        // reduce to compute time_max and time_wgt
        iter = nthreads_per_channel / 2 + nthreads_per_channel % 2;
        oddElements = nthreads_per_channel % 2;
        CMS_UNROLL_LOOP
        while (iter >= 1) {
          for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
            auto const ltx = idx.global % nthreads_per_channel;

            if (ltx < iter && !(oddElements && (ltx == iter - 1 && ltx > 0))) {
              shr_time_wgt[idx.local] += shr_time_wgt[idx.local + iter];
              shr_time_max[idx.local] += shr_time_max[idx.local + iter];
              shrTimeMax[idx.local] += shrTimeMax[idx.local + iter];
              shrTimeWgt[idx.local] += shrTimeWgt[idx.local + iter];
            }
          }

          alpaka::syncBlockThreads(acc);
          oddElements = iter % 2;
          iter = iter == 1 ? iter / 2 : iter / 2 + iter % 2;
        }

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const ltx = idx.global % nthreads_per_channel;

          // load from shared memory the 0th guy (will contain accumulated values)
          // compute
          // store into global mem
          if (ltx == 0) {
            auto const ch = idx.global / nthreads_per_channel;
            auto const tmp_time_max = shr_time_max[idx.local];
            auto const tmp_time_wgt = shr_time_wgt[idx.local];

            // we are done if there number of time ratios is 0
            if (tmp_time_wgt == 0 && tmp_time_max == 0) {
              g_state[ch] = TimeComputationState::Finished;
              continue;
            }

            // no div by 0
            auto const tMaxAlphaBeta = tmp_time_max / tmp_time_wgt;
            auto const tMaxErrorAlphaBeta = 1. / std::sqrt(tmp_time_wgt);

            tMaxAlphaBetas[ch] = tMaxAlphaBeta;
            tMaxErrorAlphaBetas[ch] = tMaxErrorAlphaBeta;
            g_accTimeMax[ch] = shrTimeMax[idx.local];
            g_accTimeWgt[ch] = shrTimeWgt[idx.local];
            g_state[ch] = TimeComputationState::NotFinished;

#ifdef DEBUG_TC_MAKERATIO
            printf("ch = %d time_max = %f time_wgt = %f\n", ch, tmp_time_max, tmp_time_wgt);
            printf("ch = %d tMaxAlphaBeta = %f tMaxErrorAlphaBeta = %f timeMax = %f timeWgt = %f\n",
                   ch,
                   tMaxAlphaBeta,
                   tMaxErrorAlphaBeta,
                   shrTimeMax[idx.local],
                   shrTimeWgt[idx.local]);
#endif
          }
        }
      }
    }
  };

  class Kernel_time_compute_findamplchi2_and_finish {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  ScalarType* const sample_values,
                                  ScalarType* const sample_value_errors,
                                  bool* const useless_samples,
                                  ScalarType* const g_tMaxAlphaBeta,
                                  ScalarType* const g_tMaxErrorAlphaBeta,
                                  ScalarType* const g_accTimeMax,
                                  ScalarType* const g_accTimeWgt,
                                  ScalarType* const sumAAsNullHypot,
                                  ScalarType* const sum0sNullHypot,
                                  ScalarType* const chi2sNullHypot,
                                  TimeComputationState* g_state,
                                  ScalarType* g_ampMaxAlphaBeta,
                                  ScalarType* g_ampMaxError,
                                  ScalarType* g_timeMax,
                                  ScalarType* g_timeError,
                                  EcalMultifitParameters const* paramsDev) const {
      /// launch ctx parameters are
      /// 10 threads per channel, N channels per block, Y blocks
      /// TODO: do we need to keep the state around or can be removed?!
      //#define DEBUG_FINDAMPLCHI2_AND_FINISH

      // constants
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;
      auto const nchannels = digisDevEB.size() + digisDevEE.size();
      auto const offsetForInputs = digisDevEB.size();

      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

      // configure shared mem
      // per block, we need #threads per block * 2 * sizeof(ScalarType)
      // we run with N channels per block
      auto* shr_sumAf = alpaka::getDynSharedMem<ScalarType>(acc);
      auto* shr_sumff = shr_sumAf + elemsPerBlock;

      for (auto gtxforward : cms::alpakatools::uniform_elements(acc, nchannels * nsamples)) {
        // go backwards through the loop to have valid values for shared variables when reading from higher element indices in serial execution
        auto gtx = nchannels * nsamples - 1 - gtxforward;
        auto const ch = gtx / nsamples;
        auto const elemIdx = gtx % elemsPerBlock;
        auto const sample = elemIdx % nsamples;

        auto const* dids = ch >= offsetForInputs ? digisDevEE.id() : digisDevEB.id();
        auto const inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;

        auto state = g_state[ch];
        auto const did = DetId{dids[inputCh]};
        auto* const amplitudeFitParameters = did.subdetId() == EcalBarrel ? paramsDev->amplitudeFitParamsEB.data()
                                                                          : paramsDev->amplitudeFitParamsEE.data();

        // TODO is that better than storing into global and launching another kernel
        // for the first 10 threads
        if (state == TimeComputationState::NotFinished) {
          auto const alpha = amplitudeFitParameters[0];
          auto const beta = amplitudeFitParameters[1];
          auto const alphabeta = alpha * beta;
          auto const invalphabeta = 1. / alphabeta;
          auto const tMaxAlphaBeta = g_tMaxAlphaBeta[ch];
          auto const sample_value = sample_values[gtx];
          auto const sample_value_error = sample_value_errors[gtx];
          auto const inverr2 =
              useless_samples[gtx] ? static_cast<ScalarType>(0) : 1. / (sample_value_error * sample_value_error);
          auto const offset = (static_cast<ScalarType>(sample) - tMaxAlphaBeta) * invalphabeta;
          auto const term1 = 1. + offset;
          auto const f = term1 > 1e-6 ? fast_expf(alpha * (fast_logf(term1) - offset)) : static_cast<ScalarType>(0.);
          auto const sumAf = sample_value * (f * inverr2);
          auto const sumff = f * (f * inverr2);

          // store into shared mem
          shr_sumAf[elemIdx] = sumAf;
          shr_sumff[elemIdx] = sumff;
        } else {
          shr_sumAf[elemIdx] = 0;
          shr_sumff[elemIdx] = 0;
        }

        alpaka::syncBlockThreads(acc);

        // reduce
        // unroll completely here (but hardcoded)
        if (sample < 5) {
          shr_sumAf[elemIdx] += shr_sumAf[elemIdx + 5];
          shr_sumff[elemIdx] += shr_sumff[elemIdx + 5];
        }

        alpaka::syncBlockThreads(acc);

        if (sample < 2) {
          // will need to subtract for ltx = 3, we double count here
          shr_sumAf[elemIdx] += shr_sumAf[elemIdx + 2] + shr_sumAf[elemIdx + 3];
          shr_sumff[elemIdx] += shr_sumff[elemIdx + 2] + shr_sumff[elemIdx + 3];
        }

        alpaka::syncBlockThreads(acc);

        if (sample == 0) {
          // exit if the state is done
          // note, we do not exit before all __synchtreads are finished
          if (state == TimeComputationState::Finished) {
            g_timeMax[ch] = 5;
            g_timeError[ch] = -999;
            continue;
          }

          // subtract to avoid double counting
          auto const sumff = shr_sumff[elemIdx] + shr_sumff[elemIdx + 1] - shr_sumff[elemIdx + 3];
          auto const sumAf = shr_sumAf[elemIdx] + shr_sumAf[elemIdx + 1] - shr_sumAf[elemIdx + 3];

          auto const ampMaxAlphaBeta = sumff > 0 ? sumAf / sumff : 0;
          auto const sumAA = sumAAsNullHypot[ch];
          auto const sum0 = sum0sNullHypot[ch];
          auto const nullChi2 = chi2sNullHypot[ch];
          if (sumff > 0) {
            auto const chi2AlphaBeta = (sumAA - sumAf * sumAf / sumff) / sum0;
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
            continue;
          }

          auto const ampMaxError = g_ampMaxError[ch];
          auto const test_ratio = ampMaxAlphaBeta / ampMaxError;
          auto const accTimeMax = g_accTimeMax[ch];
          auto const accTimeWgt = g_accTimeWgt[ch];
          auto const tMaxAlphaBeta = g_tMaxAlphaBeta[ch];
          auto const tMaxErrorAlphaBeta = g_tMaxErrorAlphaBeta[ch];
          // branch to separate large vs small pulses
          // see cpu version for more info
          if (test_ratio > 5. && accTimeWgt > 0) {
            auto const tMaxRatio = accTimeWgt > 0 ? accTimeMax / accTimeWgt : static_cast<ScalarType>(0);
            auto const tMaxErrorRatio = accTimeWgt > 0 ? 1. / std::sqrt(accTimeWgt) : static_cast<ScalarType>(0);

            if (test_ratio > 10.) {
              g_timeMax[ch] = tMaxRatio;
              g_timeError[ch] = tMaxErrorRatio;

#ifdef DEBUG_FINDAMPLCHI2_AND_FINISH
              printf("ch = %d tMaxRatio = %f tMaxErrorRatio = %f\n", ch, tMaxRatio, tMaxErrorRatio);
#endif
            } else {
              auto const timeMax = (tMaxAlphaBeta * (10. - ampMaxAlphaBeta / ampMaxError) +
                                    tMaxRatio * (ampMaxAlphaBeta / ampMaxError - 5.)) /
                                   5.;
              auto const timeError = (tMaxErrorAlphaBeta * (10. - ampMaxAlphaBeta / ampMaxError) +
                                      tMaxErrorRatio * (ampMaxAlphaBeta / ampMaxError - 5.)) /
                                     5.;
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
    }
  };

  class Kernel_time_compute_fixMGPAslew {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ScalarType* sample_values,
                                  ScalarType* sample_value_errors,
                                  bool* useless_sample_values) const {
      // constants
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;

      auto const nchannelsEB = digisDevEB.size();
      auto const offsetForInputs = nchannelsEB;

      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

      for (auto gtx : cms::alpakatools::uniform_elements(acc, nchannelsEB * nsamples)) {
        auto const elemIdx = gtx % elemsPerBlock;
        auto const sample = elemIdx % nsamples;
        auto const ch = gtx / nsamples;

        // remove thread for sample 0, oversubscribing is easier than ....
        if (sample == 0)
          continue;

        if (!use_sample(conditionsDev.sampleMask_EB(), sample))
          continue;

        int const inputGtx = ch >= offsetForInputs ? gtx - offsetForInputs * nsamples : gtx;
        auto const* digis = ch >= offsetForInputs ? digisDevEE.data()->data() : digisDevEB.data()->data();

        auto const gainIdPrev = ecalMGPA::gainId(digis[inputGtx - 1]);
        auto const gainIdNext = ecalMGPA::gainId(digis[inputGtx]);
        if (gainIdPrev >= 1 && gainIdPrev <= 3 && gainIdNext >= 1 && gainIdNext <= 3 && gainIdPrev < gainIdNext) {
          sample_values[gtx - 1] = 0;
          sample_value_errors[gtx - 1] = 1e+9;
          useless_sample_values[gtx - 1] = true;
        }
      }
    }
  };

  //#define ECAL_RECO_ALPAKA_TC_INIT_DEBUG
  class Kernel_time_computation_init {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ScalarType* sample_values,
                                  ScalarType* sample_value_errors,
                                  ScalarType* ampMaxError,
                                  bool* useless_sample_values,
                                  char* pedestal_nums) const {
      // constants
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;

      // indices
      auto const nchannelsEB = digisDevEB.size();
      auto const nchannels = nchannelsEB + digisDevEE.size();
      auto const offsetForInputs = nchannelsEB;
      auto const offsetForHashes = conditionsDev.offsetEE();

      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

      // configure shared mem
      auto* shrSampleValues = alpaka::getDynSharedMem<ScalarType>(acc);
      auto* shrSampleValueErrors = shrSampleValues + elemsPerBlock;

      for (auto txforward : cms::alpakatools::uniform_elements(acc, nchannels * nsamples)) {
        // go backwards through the loop to have valid values for shared variables when reading from higher element indices in serial execution
        auto tx = nchannels * nsamples - 1 - txforward;
        auto const ch = tx / nsamples;
        auto const elemIdx = tx % elemsPerBlock;

        int const inputTx = ch >= offsetForInputs ? tx - offsetForInputs * nsamples : tx;
        int const inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;
        auto const* digis = ch >= offsetForInputs ? digisDevEE.data()->data() : digisDevEB.data()->data();
        auto const* dids = ch >= offsetForInputs ? digisDevEE.id() : digisDevEB.id();

        // indices/inits
        auto const sample = tx % nsamples;
        auto const input_ch_start = inputCh * nsamples;
        ScalarType pedestal = 0.;
        int num = 0;

        // 0 and 1 sample values
        auto const adc0 = ecalMGPA::adc(digis[input_ch_start]);
        auto const gainId0 = ecalMGPA::gainId(digis[input_ch_start]);
        auto const adc1 = ecalMGPA::adc(digis[input_ch_start + 1]);
        auto const gainId1 = ecalMGPA::gainId(digis[input_ch_start + 1]);
        auto const did = DetId{dids[inputCh]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const sample_mask = isBarrel ? conditionsDev.sampleMask_EB() : conditionsDev.sampleMask_EE();
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

        // set pedestal
        // TODO this branch is non-divergent for a group of 10 threads
        if (gainId0 == 1 && use_sample(sample_mask, 0)) {
          pedestal = static_cast<ScalarType>(adc0);
          num = 1;

          auto const diff = adc1 - adc0;
          if (gainId1 == 1 && use_sample(sample_mask, 1) &&
              std::abs(diff) < 3 * conditionsDev.pedestals_rms_x12()[hashedId]) {
            pedestal = (pedestal + static_cast<ScalarType>(adc1)) / 2.;
            num = 2;
          }
        } else {
          pedestal = conditionsDev.pedestals_mean_x12()[ch];
        }

        // ped subtracted and gain-renormalized samples.
        auto const gainId = ecalMGPA::gainId(digis[inputTx]);
        auto const adc = ecalMGPA::adc(digis[inputTx]);

        bool bad = false;
        ScalarType sample_value, sample_value_error;
        // TODO divergent branch
        // TODO: piece below is general both for amplitudes and timing
        // potentially there is a way to reduce the amount of code...
        if (!use_sample(sample_mask, sample)) {
          bad = true;
          sample_value = 0;
          sample_value_error = 0;
        } else if (gainId == 1) {
          sample_value = static_cast<ScalarType>(adc) - pedestal;
          sample_value_error = conditionsDev.pedestals_rms_x12()[hashedId];
        } else if (gainId == 2) {
          auto const mean_x6 = conditionsDev.pedestals_mean_x6()[hashedId];
          auto const rms_x6 = conditionsDev.pedestals_rms_x6()[hashedId];
          auto const gain12Over6 = conditionsDev.gain12Over6()[hashedId];
          sample_value = (static_cast<ScalarType>(adc) - mean_x6) * gain12Over6;
          sample_value_error = rms_x6 * gain12Over6;
        } else if (gainId == 3) {
          auto const mean_x1 = conditionsDev.pedestals_mean_x1()[hashedId];
          auto const rms_x1 = conditionsDev.pedestals_rms_x1()[hashedId];
          auto const gain12Over6 = conditionsDev.gain12Over6()[hashedId];
          auto const gain6Over1 = conditionsDev.gain6Over1()[hashedId];
          sample_value = (static_cast<ScalarType>(adc) - mean_x1) * gain6Over1 * gain12Over6;
          sample_value_error = rms_x1 * gain6Over1 * gain12Over6;
        } else {
          sample_value = 0;
          sample_value_error = 0;
          bad = true;
        }

        // TODO: make sure we save things correctly when sample is useless
        auto const useless_sample = (sample_value_error <= 0) | bad;
        useless_sample_values[tx] = useless_sample;
        sample_values[tx] = sample_value;
        sample_value_errors[tx] = useless_sample ? 1e+9 : sample_value_error;

        // DEBUG
#ifdef ECAL_RECO_ALPAKA_TC_INIT_DEBUG
        if (ch == 0) {
          printf("sample = %d sample_value = %f sample_value_error = %f useless = %c\n",
                 sample,
                 sample_value,
                 sample_value_error,
                 useless_sample ? '1' : '0');
        }
#endif

        // store into the shared mem
        shrSampleValues[elemIdx] = sample_value_error > 0 ? sample_value : std::numeric_limits<ScalarType>::min();
        shrSampleValueErrors[elemIdx] = sample_value_error;
        alpaka::syncBlockThreads(acc);

        // perform the reduction with min
        if (sample < 5) {
          // note, if equal -> we keep the value with lower sample as for cpu
          shrSampleValueErrors[elemIdx] = shrSampleValues[elemIdx] < shrSampleValues[elemIdx + 5]
                                              ? shrSampleValueErrors[elemIdx + 5]
                                              : shrSampleValueErrors[elemIdx];
          shrSampleValues[elemIdx] = std::max(shrSampleValues[elemIdx], shrSampleValues[elemIdx + 5]);
        }
        alpaka::syncBlockThreads(acc);

        // a bit of an overkill, but easier than to compare across 3 values
        if (sample < 3) {
          shrSampleValueErrors[elemIdx] = shrSampleValues[elemIdx] < shrSampleValues[elemIdx + 3]
                                              ? shrSampleValueErrors[elemIdx + 3]
                                              : shrSampleValueErrors[elemIdx];
          shrSampleValues[elemIdx] = std::max(shrSampleValues[elemIdx], shrSampleValues[elemIdx + 3]);
        }
        alpaka::syncBlockThreads(acc);

        if (sample < 2) {
          shrSampleValueErrors[elemIdx] = shrSampleValues[elemIdx] < shrSampleValues[elemIdx + 2]
                                              ? shrSampleValueErrors[elemIdx + 2]
                                              : shrSampleValueErrors[elemIdx];
          shrSampleValues[elemIdx] = std::max(shrSampleValues[elemIdx], shrSampleValues[elemIdx + 2]);
        }
        alpaka::syncBlockThreads(acc);

        if (sample == 0) {
          // we only need the max error
          auto const maxSampleValueError = shrSampleValues[elemIdx] < shrSampleValues[elemIdx + 1]
                                               ? shrSampleValueErrors[elemIdx + 1]
                                               : shrSampleValueErrors[elemIdx];

          // # pedestal samples used
          pedestal_nums[ch] = num;
          // this is used downstream
          ampMaxError[ch] = maxSampleValueError;

          // DEBUG
#ifdef ECAL_RECO_ALPAKA_TC_INIT_DEBUG
          if (ch == 0) {
            printf("pedestal_nums = %d ampMaxError = %f\n", num, maxSampleValueError);
          }
#endif
        }
      }
    }
  };

  ///
  /// launch context parameters: 1 thread per channel
  ///
  //#define DEBUG_TIME_CORRECTION
  class Kernel_time_correction_and_finalize {
    using ScalarType = ::ecal::multifit::SampleVector::Scalar;

  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  EcalUncalibratedRecHitDeviceCollection::View uncalibRecHitsEB,
                                  EcalUncalibratedRecHitDeviceCollection::View uncalibRecHitsEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ScalarType* const g_timeMax,
                                  ScalarType* const g_timeError,
                                  ConfigurationParameters::type const timeConstantTermEB,
                                  ConfigurationParameters::type const timeConstantTermEE,
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
                                  ConfigurationParameters::type const outOfTimeThreshG61mEE) const {
      // constants
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;
      auto const nchannelsEB = digisDevEB.size();
      auto const nchannels = nchannelsEB + digisDevEE.size();
      auto const offsetForInputs = nchannelsEB;
      auto const offsetForHashes = conditionsDev.offsetEE();

      for (auto gtx : cms::alpakatools::uniform_elements(acc, nchannels)) {
        const int inputGtx = gtx >= offsetForInputs ? gtx - offsetForInputs : gtx;
        auto const* dids = gtx >= offsetForInputs ? digisDevEE.id() : digisDevEB.id();
        auto const* digis = gtx >= offsetForInputs ? digisDevEE.data()->data() : digisDevEB.data()->data();

        auto* g_amplitude = gtx >= nchannelsEB ? uncalibRecHitsEE.amplitude() : uncalibRecHitsEB.amplitude();
        auto* g_jitter = gtx >= nchannelsEB ? uncalibRecHitsEE.jitter() : uncalibRecHitsEB.jitter();
        auto* g_jitterError = gtx >= nchannelsEB ? uncalibRecHitsEE.jitterError() : uncalibRecHitsEB.jitterError();
        auto* flags = gtx >= nchannelsEB ? uncalibRecHitsEE.flags() : uncalibRecHitsEB.flags();

        auto const did = DetId{dids[inputGtx]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());
        // need to access the underlying data directly here because the std::arrays have different size for EB and EE, which is not compatible with the ? operator
        auto* const amplitudeBins = isBarrel ? conditionsDev.timeBiasCorrections_amplitude_EB().data()
                                             : conditionsDev.timeBiasCorrections_amplitude_EE().data();
        auto* const shiftBins = isBarrel ? conditionsDev.timeBiasCorrections_shift_EB().data()
                                         : conditionsDev.timeBiasCorrections_shift_EE().data();
        auto const amplitudeBinsSize =
            isBarrel ? conditionsDev.timeBiasCorrectionSizeEB() : conditionsDev.timeBiasCorrectionSizeEE();
        auto const timeConstantTerm = isBarrel ? timeConstantTermEB : timeConstantTermEE;
        auto const timeNconst = isBarrel ? timeNconstEB : timeNconstEE;
        auto const offsetTimeValue = isBarrel ? conditionsDev.timeOffset_EB() : conditionsDev.timeOffset_EE();
        auto const amplitudeThreshold = isBarrel ? amplitudeThresholdEB : amplitudeThresholdEE;
        auto const outOfTimeThreshG12p = isBarrel ? outOfTimeThreshG12pEB : outOfTimeThreshG12pEE;
        auto const outOfTimeThreshG12m = isBarrel ? outOfTimeThreshG12mEB : outOfTimeThreshG12mEE;
        auto const outOfTimeThreshG61p = isBarrel ? outOfTimeThreshG61pEB : outOfTimeThreshG61pEE;
        auto const outOfTimeThreshG61m = isBarrel ? outOfTimeThreshG61mEB : outOfTimeThreshG61mEE;

        // load some
        auto const amplitude = g_amplitude[inputGtx];
        auto const rms_x12 = conditionsDev.pedestals_rms_x12()[hashedId];
        auto const timeCalibConst = conditionsDev.timeCalibConstants()[hashedId];

        int myBin = -1;
        for (size_t bin = 0; bin < amplitudeBinsSize; ++bin) {
          if (amplitude > amplitudeBins[bin])
            myBin = bin;
          else
            break;
        }

        ScalarType correction = 0;
        if (myBin == -1) {
          correction = shiftBins[0];
        } else if (myBin == static_cast<int>(amplitudeBinsSize) - 1) {
          correction = shiftBins[myBin];
        } else {
          correction = shiftBins[myBin + 1] - shiftBins[myBin];
          correction *= (amplitude - amplitudeBins[myBin]) / (amplitudeBins[myBin + 1] - amplitudeBins[myBin]);
          correction += shiftBins[myBin];
        }

        // correction * 1./25.
        correction = correction * 0.04;
        auto const timeMax = g_timeMax[gtx];
        auto const timeError = g_timeError[gtx];
        auto const jitter = timeMax - 5 + correction;
        auto const jitterError =
            std::sqrt(timeError * timeError + timeConstantTerm * timeConstantTerm * 0.04 * 0.04);  // 0.04 = 1./25.

#ifdef DEBUG_TIME_CORRECTION
        printf("ch = %d timeMax = %f timeError = %f jitter = %f correction = %f\n",
               gtx,
               timeMax,
               timeError,
               jitter,
               correction);
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
              auto const gainid = ecalMGPA::gainId(digis[nsamples * inputGtx + isample]);
              if (gainid != 1) {
                threshP = outOfTimeThreshG61p;
                threshM = outOfTimeThreshG61m;
                break;
              }
            }
          }

          auto const correctedTime = (timeMax - 5) * 25 + timeCalibConst + offsetTimeValue;
          auto const nterm = timeNconst * rms_x12 / amplitude;
          auto const sigmat = std::sqrt(nterm * nterm + timeConstantTerm * timeConstantTerm);
          if (correctedTime > sigmat * threshP || correctedTime < -sigmat * threshM)
            flags[inputGtx] |= 0x1 << EcalUncalibratedRecHit::kOutOfTime;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

namespace alpaka::trait {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit;

  //! The trait for getting the size of the block shared dynamic memory for Kernel_time_compute_nullhypot.
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_time_compute_nullhypot, TAcc> {
    //! \return The size of the shared memory allocated for a block.
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_time_compute_nullhypot const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      using ScalarType = ecal::multifit::SampleVector::Scalar;

      // return the amount of dynamic shared memory needed
      std::size_t bytes = threadsPerBlock[0u] * elemsPerThread[0u] * 4 * sizeof(ScalarType);
      return bytes;
    }
  };

  //! The trait for getting the size of the block shared dynamic memory for Kernel_time_compute_makeratio.
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_time_compute_makeratio, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_time_compute_makeratio const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      using ScalarType = ecal::multifit::SampleVector::Scalar;

      std::size_t bytes = (8 * sizeof(ScalarType) + 3 * sizeof(bool)) * threadsPerBlock[0u] * elemsPerThread[0u];
      return bytes;
    }
  };

  //! The trait for getting the size of the block shared dynamic memory for Kernel_time_compute_findamplchi2_and_finish.
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_time_compute_findamplchi2_and_finish, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_time_compute_findamplchi2_and_finish const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      using ScalarType = ecal::multifit::SampleVector::Scalar;

      std::size_t bytes = 2 * threadsPerBlock[0u] * elemsPerThread[0u] * sizeof(ScalarType);
      return bytes;
    }
  };

  //! The trait for getting the size of the block shared dynamic memory for Kernel_time_computation_init.
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_time_computation_init, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_time_computation_init const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      using ScalarType = ecal::multifit::SampleVector::Scalar;

      std::size_t bytes = 2 * threadsPerBlock[0u] * elemsPerThread[0u] * sizeof(ScalarType);
      return bytes;
    }
  };

}  // namespace alpaka::trait

#endif  // RecoLocalCalo_EcalRecProducers_plugins_TimeComputationKernels_h
