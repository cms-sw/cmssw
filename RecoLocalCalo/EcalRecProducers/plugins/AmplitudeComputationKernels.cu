#include <cmath>
#include <limits>

#include <cuda.h>

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernels.h"
#include "KernelHelpers.h"

namespace ecal {
  namespace multifit {

    template <typename MatrixType>
    __device__ __forceinline__ bool update_covariance(EcalPulseCovariance const& pulse_covariance,
                                                      MatrixType& inverse_cov,
                                                      SampleVector const& amplitudes) {
      constexpr int nsamples = SampleVector::RowsAtCompileTime;
      constexpr int npulses = BXVectorType::RowsAtCompileTime;

      CMS_UNROLL_LOOP
      for (unsigned int ipulse = 0; ipulse < npulses; ipulse++) {
        auto const amplitude = amplitudes.coeff(ipulse);
        if (amplitude == 0)
          continue;

        // FIXME: ipulse - 5 -> ipulse - firstOffset
        int bx = ipulse - 5;
        int first_sample_t = std::max(0, bx + 3);
        int offset = -3 - bx;

        auto const value_sq = amplitude * amplitude;

        for (int col = first_sample_t; col < nsamples; col++) {
          for (int row = col; row < nsamples; row++) {
            inverse_cov(row, col) += value_sq * __ldg(&pulse_covariance.covval[row + offset][col + offset]);
          }
        }
      }

      return true;
    }

    ///
    /// launch ctx parameters are (nchannels / block, blocks)
    /// TODO: trivial impl for now, there must be a way to improve
    ///
    /// Conventions:
    ///   - amplitudes -> solution vector, what we are fitting for
    ///   - samples -> raw detector responses
    ///   - passive constraint - satisfied constraint
    ///   - active constraint - unsatisfied (yet) constraint
    ///
    __global__ void kernel_minimize(uint32_t const* dids_eb,
                                    uint32_t const* dids_ee,
                                    SampleMatrix const* __restrict__ noisecov,
                                    EcalPulseCovariance const* __restrict__ pulse_covariance,
                                    BXVectorType* bxs,
                                    SampleVector const* __restrict__ samples,
                                    SampleVector* amplitudesEB,
                                    SampleVector* amplitudesEE,
                                    PulseMatrixType const* __restrict__ pulse_matrix,
                                    ::ecal::reco::StorageScalarType* chi2sEB,
                                    ::ecal::reco::StorageScalarType* chi2sEE,
                                    ::ecal::reco::StorageScalarType* energiesEB,
                                    ::ecal::reco::StorageScalarType* energiesEE,
                                    char* acState,
                                    int nchannels,
                                    int max_iterations,
                                    uint32_t const offsetForHashes,
                                    uint32_t const offsetForInputs) {
      // FIXME: ecal has 10 samples and 10 pulses....
      // but this needs to be properly treated and renamed everywhere
      constexpr auto NSAMPLES = SampleMatrix::RowsAtCompileTime;
      constexpr auto NPULSES = SampleMatrix::ColsAtCompileTime;
      static_assert(NSAMPLES == NPULSES);

      using DataType = SampleVector::Scalar;

      extern __shared__ char shrmem[];
      DataType* shrMatrixLForFnnlsStorage =
          reinterpret_cast<DataType*>(shrmem) + calo::multifit::MapSymM<DataType, NPULSES>::total * threadIdx.x;
      DataType* shrAtAStorage = reinterpret_cast<DataType*>(shrmem) +
                                calo::multifit::MapSymM<DataType, NPULSES>::total * (threadIdx.x + blockDim.x);

      // channel
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

// ref the right ptr
#define ARRANGE(var) auto* var = idx >= offsetForInputs ? var##EE : var##EB
      ARRANGE(amplitudes);
      ARRANGE(chi2s);
      ARRANGE(energies);
#undef ARRANGE

      if (idx < nchannels) {
        if (static_cast<MinimizationState>(acState[idx]) == MinimizationState::Precomputed)
          return;

        // get the hash
        int const inputCh = idx >= offsetForInputs ? idx - offsetForInputs : idx;
        auto const* dids = idx >= offsetForInputs ? dids_ee : dids_eb;
        auto const did = DetId{dids[inputCh]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

        // inits
        int iter = 0;
        int npassive = 0;

        calo::multifit::ColumnVector<NPULSES, int> pulseOffsets;
        CMS_UNROLL_LOOP
        for (int i = 0; i < NPULSES; ++i)
          pulseOffsets(i) = i;

        calo::multifit::ColumnVector<NPULSES, DataType> resultAmplitudes;
        CMS_UNROLL_LOOP
        for (int counter = 0; counter < NPULSES; counter++)
          resultAmplitudes(counter) = 0;

        // inits
        //SampleDecompLLT covariance_decomposition;
        //SampleMatrix inverse_cov;
        //        SampleVector::Scalar chi2 = 0, chi2_now = 0;
        float chi2 = 0, chi2_now = 0;

        // loop until ocnverge
        while (true) {
          if (iter >= max_iterations)
            break;

          //inverse_cov = noisecov[idx];
          //DataType covMatrixStorage[MapSymM<DataType, NSAMPLES>::total];
          DataType* covMatrixStorage = shrMatrixLForFnnlsStorage;
          calo::multifit::MapSymM<DataType, NSAMPLES> covMatrix{covMatrixStorage};
          int counter = 0;
          CMS_UNROLL_LOOP
          for (int col = 0; col < NSAMPLES; col++) {
            CMS_UNROLL_LOOP
            for (int row = col; row < NSAMPLES; row++)
              covMatrixStorage[counter++] = __ldg(&noisecov[idx].coeffRef(row, col));
          }
          update_covariance(pulse_covariance[hashedId], covMatrix, resultAmplitudes);

          // compute actual covariance decomposition
          //covariance_decomposition.compute(inverse_cov);
          //auto const& matrixL = covariance_decomposition.matrixL();
          DataType matrixLStorage[calo::multifit::MapSymM<DataType, NSAMPLES>::total];
          calo::multifit::MapSymM<DataType, NSAMPLES> matrixL{matrixLStorage};
          calo::multifit::compute_decomposition_unrolled(matrixL, covMatrix);

          // L * A = P
          calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES> A;
          calo::multifit::solve_forward_subst_matrix(A, pulse_matrix[idx], matrixL);

          // L b = s
          float reg_b[NSAMPLES];
          calo::multifit::solve_forward_subst_vector(reg_b, samples[idx], matrixL);

          // FIXME: shared mem
          //DataType AtAStorage[MapSymM<DataType, NPULSES>::total];
          calo::multifit::MapSymM<DataType, NPULSES> AtA{shrAtAStorage};
          //SampleMatrix AtA;
          SampleVector Atb;
          CMS_UNROLL_LOOP
          for (int icol = 0; icol < NPULSES; icol++) {
            float reg_ai[NSAMPLES];

            // load column icol
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; counter++)
              reg_ai[counter] = A(counter, icol);

            // compute diagoanl
            float sum = 0.f;
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum += reg_ai[counter] * reg_ai[counter];

            // store
            AtA(icol, icol) = sum;

            // go thru the other columns
            CMS_UNROLL_LOOP
            for (int j = icol + 1; j < NPULSES; j++) {
              // load column j
              float reg_aj[NSAMPLES];
              CMS_UNROLL_LOOP
              for (int counter = 0; counter < NSAMPLES; counter++)
                reg_aj[counter] = A(counter, j);

              // accum
              float sum = 0.f;
              CMS_UNROLL_LOOP
              for (int counter = 0; counter < NSAMPLES; counter++)
                sum += reg_aj[counter] * reg_ai[counter];

              // store
              //AtA(icol, j) = sum;
              AtA(j, icol) = sum;
            }

            // Atb accum
            float sum_atb = 0.f;
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum_atb += reg_ai[counter] * reg_b[counter];

            // store atb
            Atb(icol) = sum_atb;
          }

          // FIXME: shared mem
          //DataType matrixLForFnnlsStorage[MapSymM<DataType, NPULSES>::total];
          calo::multifit::MapSymM<DataType, NPULSES> matrixLForFnnls{shrMatrixLForFnnlsStorage};

          calo::multifit::fnnls(AtA,
                                Atb,
                                //amplitudes[idx],
                                resultAmplitudes,
                                npassive,
                                pulseOffsets,
                                matrixLForFnnls,
                                1e-11,
                                500,
                                16,
                                2);

          calo::multifit::calculateChiSq(matrixL, pulse_matrix[idx], resultAmplitudes, samples[idx], chi2_now);

          auto deltachi2 = chi2_now - chi2;
          chi2 = chi2_now;

          if (std::abs(deltachi2) < 1e-3)
            break;

          //---- AM: TEST
          //---- it was 3 lines above, now here as in the CPU version
          ++iter;
        }

        // store to global output values
        // FIXME: amplitudes are used in global directly
        chi2s[inputCh] = chi2;
        energies[inputCh] = resultAmplitudes(5);

        CMS_UNROLL_LOOP
        for (int counter = 0; counter < NPULSES; counter++)
          amplitudes[inputCh](counter) = resultAmplitudes(counter);
      }
    }

    namespace v1 {

      void minimization_procedure(EventInputDataGPU const& eventInputGPU,
                                  EventOutputDataGPU& eventOutputGPU,
                                  EventDataForScratchGPU& scratch,
                                  ConditionsProducts const& conditions,
                                  ConfigurationParameters const& configParameters,
                                  cudaStream_t cudaStream) {
        using DataType = SampleVector::Scalar;
        unsigned int totalChannels = eventInputGPU.ebDigis.size + eventInputGPU.eeDigis.size;
        //    unsigned int threads_min = conf.threads.x;
        // TODO: configure from python
        unsigned int threads_min = configParameters.kernelMinimizeThreads[0];
        unsigned int blocks_min = threads_min > totalChannels ? 1 : (totalChannels + threads_min - 1) / threads_min;
        uint32_t const offsetForHashes = conditions.offsetForHashes;
        uint32_t const offsetForInputs = eventInputGPU.ebDigis.size;
        auto const nbytesShared = 2 * threads_min *
                                  calo::multifit::MapSymM<DataType, SampleVector::RowsAtCompileTime>::total *
                                  sizeof(DataType);
        kernel_minimize<<<blocks_min, threads_min, nbytesShared, cudaStream>>>(
            eventInputGPU.ebDigis.ids.get(),
            eventInputGPU.eeDigis.ids.get(),
            (SampleMatrix*)scratch.noisecov.get(),
            conditions.pulseCovariances.values,
            (BXVectorType*)scratch.activeBXs.get(),
            (SampleVector*)scratch.samples.get(),
            (SampleVector*)eventOutputGPU.recHitsEB.amplitudesAll.get(),
            (SampleVector*)eventOutputGPU.recHitsEE.amplitudesAll.get(),
            (PulseMatrixType*)scratch.pulse_matrix.get(),
            eventOutputGPU.recHitsEB.chi2.get(),
            eventOutputGPU.recHitsEE.chi2.get(),
            eventOutputGPU.recHitsEB.amplitude.get(),
            eventOutputGPU.recHitsEE.amplitude.get(),
            scratch.acState.get(),
            totalChannels,
            50,
            offsetForHashes,
            offsetForInputs);
        cudaCheck(cudaGetLastError());
      }

    }  // namespace v1

  }  // namespace multifit
}  // namespace ecal
