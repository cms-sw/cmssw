#include <iostream>
#include <limits>

#include <cuda.h>

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernels.h"
#include "KernelHelpers.h"

namespace ecal {
  namespace multifit {

    void eigen_solve_submatrix(SampleMatrix& mat, SampleVector& invec, SampleVector& outvec, unsigned NP) {
      using namespace Eigen;
      switch (NP) {  // pulse matrix is always square.
        case 10: {
          Matrix<SampleMatrix::Scalar, 10, 10> temp = mat.topLeftCorner<10, 10>();
          outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
          break;
        }
        case 9: {
          Matrix<SampleMatrix::Scalar, 9, 9> temp = mat.topLeftCorner<9, 9>();
          outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
          break;
        }
        case 8: {
          Matrix<SampleMatrix::Scalar, 8, 8> temp = mat.topLeftCorner<8, 8>();
          outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
          break;
        }
        case 7: {
          Matrix<SampleMatrix::Scalar, 7, 7> temp = mat.topLeftCorner<7, 7>();
          outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
          break;
        }
        case 6: {
          Matrix<SampleMatrix::Scalar, 6, 6> temp = mat.topLeftCorner<6, 6>();
          outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
          break;
        }
        case 5: {
          Matrix<SampleMatrix::Scalar, 5, 5> temp = mat.topLeftCorner<5, 5>();
          outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
          break;
        }
        case 4: {
          Matrix<SampleMatrix::Scalar, 4, 4> temp = mat.topLeftCorner<4, 4>();
          outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
          break;
        }
        case 3: {
          Matrix<SampleMatrix::Scalar, 3, 3> temp = mat.topLeftCorner<3, 3>();
          outvec.head<3>() = temp.ldlt().solve(invec.head<3>());
          break;
        }
        case 2: {
          Matrix<SampleMatrix::Scalar, 2, 2> temp = mat.topLeftCorner<2, 2>();
          outvec.head<2>() = temp.ldlt().solve(invec.head<2>());
          break;
        }
        case 1: {
          Matrix<SampleMatrix::Scalar, 1, 1> temp = mat.topLeftCorner<1, 1>();
          outvec.head<1>() = temp.ldlt().solve(invec.head<1>());
          break;
        }
        default:
          return;
      }
    }

    template <typename MatrixType>
    __device__ __forceinline__ bool update_covariance(EcalPulseCovariance const& pulse_covariance,
                                                      MatrixType& inverse_cov,
                                                      SampleVector const& amplitudes) {
      constexpr int nsamples = SampleVector::RowsAtCompileTime;
      constexpr int npulses = BXVectorType::RowsAtCompileTime;

#pragma unroll
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

      // FIXME: remove eitehr idx or ch -> they are teh same thing
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

// ref the right ptr
#define ARRANGE(var) auto* var = idx >= offsetForInputs ? var##EE : var##EB
      ARRANGE(amplitudes);
      ARRANGE(chi2s);
      ARRANGE(energies);
#undef ARRANGE

      auto const ch = idx;
      if (idx < nchannels) {
        if (static_cast<MinimizationState>(acState[idx]) == MinimizationState::Precomputed)
          return;

        // get the hash
        int const inputCh = ch >= offsetForInputs ? ch - offsetForInputs : ch;
        auto const* dids = ch >= offsetForInputs ? dids_ee : dids_eb;
        auto const did = DetId{dids[inputCh]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

        // inits
        int iter = 0;
        int npassive = 0;

        calo::multifit::ColumnVector<NPULSES, int> pulseOffsets;
#pragma unroll
        for (int i = 0; i < NPULSES; ++i)
          pulseOffsets(i) = i;

        calo::multifit::ColumnVector<NPULSES, DataType> resultAmplitudes;
#pragma unroll
        for (int counter = 0; counter < NPULSES; counter++)
          resultAmplitudes(counter) = 0;

        // inits
        //SampleDecompLLT covariance_decomposition;
        //SampleMatrix inverse_cov;
        SampleVector::Scalar chi2 = 0, chi2_now = 0;

        // loop until ocnverge
        while (true) {
          if (iter >= max_iterations)
            break;

          //inverse_cov = noisecov[idx];
          //DataType covMatrixStorage[MapSymM<DataType, NSAMPLES>::total];
          DataType* covMatrixStorage = shrMatrixLForFnnlsStorage;
          calo::multifit::MapSymM<DataType, NSAMPLES> covMatrix{covMatrixStorage};
          int counter = 0;
#pragma unroll
          for (int col = 0; col < NSAMPLES; col++)
#pragma unroll
            for (int row = col; row < NSAMPLES; row++)
              covMatrixStorage[counter++] = __ldg(&noisecov[idx].coeffRef(row, col));

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
#pragma unroll
          for (int icol = 0; icol < NPULSES; icol++) {
            float reg_ai[NSAMPLES];

// load column icol
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              reg_ai[counter] = A(counter, icol);

            // compute diagoanl
            float sum = 0.f;
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum += reg_ai[counter] * reg_ai[counter];

            // store
            AtA(icol, icol) = sum;

// go thru the other columns
#pragma unroll
            for (int j = icol + 1; j < NPULSES; j++) {
              // load column j
              float reg_aj[NSAMPLES];
#pragma unroll
              for (int counter = 0; counter < NSAMPLES; counter++)
                reg_aj[counter] = A(counter, j);

              // accum
              float sum = 0.f;
#pragma unroll
              for (int counter = 0; counter < NSAMPLES; counter++)
                sum += reg_aj[counter] * reg_ai[counter];

              // store
              //AtA(icol, j) = sum;
              AtA(j, icol) = sum;
            }

            // Atb accum
            float sum_atb = 0.f;
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum_atb += reg_ai[counter] * reg_b[counter];

            // store atb
            Atb(icol) = sum_atb;
          }

          // FIXME: shared mem
          //DataType matrixLForFnnlsStorage[MapSymM<DataType, NPULSES>::total];
          calo::multifit::MapSymM<DataType, NPULSES> matrixLForFnnls{shrMatrixLForFnnlsStorage};

          fnnls(AtA,
                Atb,
                //amplitudes[idx],
                resultAmplitudes,
                npassive,
                pulseOffsets,
                matrixLForFnnls,
                1e-11,
                500);

          {
            DataType accum[NSAMPLES];
// load accum
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              accum[counter] = -samples[idx](counter);

            // iterate
            for (int icol = 0; icol < NPULSES; icol++) {
              DataType pm_col[NSAMPLES];

// preload a column of pulse matrix
#pragma unroll
              for (int counter = 0; counter < NSAMPLES; counter++)
                pm_col[counter] = __ldg(&pulse_matrix[idx].coeffRef(counter, icol));

// accum
#pragma unroll
              for (int counter = 0; counter < NSAMPLES; counter++)
                accum[counter] += resultAmplitudes[icol] * pm_col[counter];
            }

            DataType reg_L[NSAMPLES];
            DataType accumSum = 0;

// preload a column and load column 0 of cholesky
#pragma unroll
            for (int i = 0; i < NSAMPLES; i++)
              reg_L[i] = matrixL(i, 0);

            // compute x0 and store it
            auto x_prev = accum[0] / reg_L[0];
            accumSum += x_prev * x_prev;

// iterate
#pragma unroll
            for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
              for (int counter = iL; counter < NSAMPLES; counter++)
                accum[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
              for (int counter = iL; counter < NSAMPLES; counter++)
                reg_L[counter] = matrixL(counter, iL);

              // compute the next x for M(iL, icol)
              x_prev = accum[iL] / reg_L[iL];

              // store teh result value
              accumSum += x_prev * x_prev;
            }

            chi2_now = accumSum;
          }

          auto deltachi2 = chi2_now - chi2;
          chi2 = chi2_now;

          if (ecal::abs(deltachi2) < 1e-3)
            break;

          //---- AM: TEST
          //---- it was 3 lines above, now here as in the CPU version
          ++iter;
        }

        // store to global output values
        // FIXME: amplitudes are used in global directly
        chi2s[inputCh] = chi2;
        energies[inputCh] = resultAmplitudes(5);

#pragma unroll
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
