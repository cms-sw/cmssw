#include <cmath>
#include <limits>
#include <alpaka/alpaka.hpp>

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "DataFormats/CaloRecHit/interface/MultifitComputations.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "AmplitudeComputationKernels.h"
#include "KernelHelpers.h"
#include "EcalUncalibRecHitMultiFitAlgoPortable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  using namespace ::ecal::multifit;

  template <typename MatrixType>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void update_covariance(EcalPulseCovariance const& pulse_covariance,
                                                        MatrixType& inverse_cov,
                                                        SampleVector const& amplitudes) {
    constexpr auto nsamples = SampleVector::RowsAtCompileTime;
    constexpr auto npulses = BXVectorType::RowsAtCompileTime;

    CMS_UNROLL_LOOP
    for (unsigned int ipulse = 0; ipulse < npulses; ++ipulse) {
      auto const amplitude = amplitudes.coeff(ipulse);
      if (amplitude == 0)
        continue;

      // FIXME: ipulse - 5 -> ipulse - firstOffset
      int bx = ipulse - 5;
      int first_sample_t = std::max(0, bx + 3);
      int offset = -3 - bx;

      auto const value_sq = amplitude * amplitude;

      for (int col = first_sample_t; col < nsamples; ++col) {
        for (int row = col; row < nsamples; ++row) {
          inverse_cov(row, col) += value_sq * pulse_covariance.covval[row + offset][col + offset];
        }
      }
    }
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
  class Kernel_minimize {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  InputProduct::ConstView const& digisDevEB,
                                  InputProduct::ConstView const& digisDevEE,
                                  OutputProduct::View uncalibRecHitsEB,
                                  OutputProduct::View uncalibRecHitsEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ::ecal::multifit::SampleMatrix const* noisecov,
                                  ::ecal::multifit::PulseMatrixType const* pulse_matrix,
                                  ::ecal::multifit::BXVectorType* bxs,
                                  ::ecal::multifit::SampleVector const* samples,
                                  bool* hasSwitchToGain6,
                                  bool* hasSwitchToGain1,
                                  bool* isSaturated,
                                  char* acState,
                                  int max_iterations) const {
      // FIXME: ecal has 10 samples and 10 pulses....
      // but this needs to be properly treated and renamed everywhere
      constexpr auto NSAMPLES = SampleMatrix::RowsAtCompileTime;
      constexpr auto NPULSES = SampleMatrix::ColsAtCompileTime;
      static_assert(NSAMPLES == NPULSES);

      using DataType = SampleVector::Scalar;

      auto const elemsPerBlock(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      auto const nchannelsEB = digisDevEB.size();
      auto const nchannels = nchannelsEB + digisDevEE.size();
      auto const offsetForHashes = conditionsDev.offsetEE();

      auto const* pulse_covariance = reinterpret_cast<const EcalPulseCovariance*>(conditionsDev.pulseCovariance());

      // shared memory
      DataType* shrmem = alpaka::getDynSharedMem<DataType>(acc);

      // channel
      for (auto idx : cms::alpakatools::uniform_elements(acc, nchannels)) {
        if (static_cast<MinimizationState>(acState[idx]) == MinimizationState::Precomputed)
          continue;

        auto const elemIdx = idx % elemsPerBlock;

        // shared memory pointers
        DataType* shrMatrixLForFnnlsStorage = shrmem + calo::multifit::MapSymM<DataType, NPULSES>::total * elemIdx;
        DataType* shrAtAStorage =
            shrmem + calo::multifit::MapSymM<DataType, NPULSES>::total * (elemIdx + elemsPerBlock);

        auto* amplitudes =
            reinterpret_cast<SampleVector*>(idx >= nchannelsEB ? uncalibRecHitsEE.outOfTimeAmplitudes()->data()
                                                               : uncalibRecHitsEB.outOfTimeAmplitudes()->data());
        auto* energies = idx >= nchannelsEB ? uncalibRecHitsEE.amplitude() : uncalibRecHitsEB.amplitude();
        auto* chi2s = idx >= nchannelsEB ? uncalibRecHitsEE.chi2() : uncalibRecHitsEB.chi2();

        // get the hash
        int const inputCh = idx >= nchannelsEB ? idx - nchannelsEB : idx;
        auto const* dids = idx >= nchannelsEB ? digisDevEE.id() : digisDevEB.id();
        auto const did = DetId{dids[inputCh]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

        // inits
        int npassive = 0;

        calo::multifit::ColumnVector<NPULSES, int> pulseOffsets;
        CMS_UNROLL_LOOP
        for (int i = 0; i < NPULSES; ++i)
          pulseOffsets(i) = i;

        calo::multifit::ColumnVector<NPULSES, DataType> resultAmplitudes;
        CMS_UNROLL_LOOP
        for (int counter = 0; counter < NPULSES; ++counter)
          resultAmplitudes(counter) = 0;

        // inits
        //SampleDecompLLT covariance_decomposition;
        //SampleMatrix inverse_cov;
        //        SampleVector::Scalar chi2 = 0, chi2_now = 0;
        float chi2 = 0, chi2_now = 0;

        // loop for up to max_iterations
        for (int iter = 0; iter < max_iterations; ++iter) {
          //inverse_cov = noisecov[idx];
          //DataType covMatrixStorage[MapSymM<DataType, NSAMPLES>::total];
          DataType* covMatrixStorage = shrMatrixLForFnnlsStorage;
          calo::multifit::MapSymM<DataType, NSAMPLES> covMatrix{covMatrixStorage};
          int counter = 0;
          CMS_UNROLL_LOOP
          for (int col = 0; col < NSAMPLES; ++col) {
            CMS_UNROLL_LOOP
            for (int row = col; row < NSAMPLES; ++row) {
              covMatrixStorage[counter++] = noisecov[idx].coeffRef(row, col);
            }
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
          for (int icol = 0; icol < NPULSES; ++icol) {
            float reg_ai[NSAMPLES];

            // load column icol
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; ++counter)
              reg_ai[counter] = A(counter, icol);

            // compute diagoanl
            float sum = 0.f;
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; ++counter)
              sum += reg_ai[counter] * reg_ai[counter];

            // store
            AtA(icol, icol) = sum;

            // go thru the other columns
            CMS_UNROLL_LOOP
            for (int j = icol + 1; j < NPULSES; ++j) {
              // load column j
              float reg_aj[NSAMPLES];
              CMS_UNROLL_LOOP
              for (int counter = 0; counter < NSAMPLES; ++counter)
                reg_aj[counter] = A(counter, j);

              // accum
              float sum = 0.f;
              CMS_UNROLL_LOOP
              for (int counter = 0; counter < NSAMPLES; ++counter)
                sum += reg_aj[counter] * reg_ai[counter];

              // store
              //AtA(icol, j) = sum;
              AtA(j, icol) = sum;
            }

            // Atb accum
            float sum_atb = 0.f;
            CMS_UNROLL_LOOP
            for (int counter = 0; counter < NSAMPLES; ++counter)
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

          auto const deltachi2 = chi2_now - chi2;
          chi2 = chi2_now;

          if (std::abs(deltachi2) < 1e-3)
            break;
        }

        // store to global output values
        // FIXME: amplitudes are used in global directly
        chi2s[inputCh] = chi2;
        energies[inputCh] = resultAmplitudes(5);

        CMS_UNROLL_LOOP
        for (int counter = 0; counter < NPULSES; ++counter)
          amplitudes[inputCh](counter) = resultAmplitudes(counter);
      }
    }
  };

  void minimization_procedure(Queue& queue,
                              InputProduct const& digisDevEB,
                              InputProduct const& digisDevEE,
                              OutputProduct& uncalibRecHitsDevEB,
                              OutputProduct& uncalibRecHitsDevEE,
                              EventDataForScratchDevice& scratch,
                              EcalMultifitConditionsDevice const& conditionsDev,
                              ConfigurationParameters const& configParams,
                              uint32_t const totalChannels) {
    using DataType = SampleVector::Scalar;
    // TODO: configure from python
    auto threads_min = configParams.kernelMinimizeThreads[0];
    auto blocks_min = cms::alpakatools::divide_up_by(totalChannels, threads_min);

    auto workDivMinimize = cms::alpakatools::make_workdiv<Acc1D>(blocks_min, threads_min);
    alpaka::exec<Acc1D>(queue,
                        workDivMinimize,
                        Kernel_minimize{},
                        digisDevEB.const_view(),
                        digisDevEE.const_view(),
                        uncalibRecHitsDevEB.view(),
                        uncalibRecHitsDevEE.view(),
                        conditionsDev.const_view(),
                        reinterpret_cast<::ecal::multifit::SampleMatrix*>(scratch.noisecovDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::PulseMatrixType*>(scratch.pulse_matrixDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::BXVectorType*>(scratch.activeBXsDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::SampleVector*>(scratch.samplesDevBuf.data()),
                        scratch.hasSwitchToGain6DevBuf.data(),
                        scratch.hasSwitchToGain1DevBuf.data(),
                        scratch.isSaturatedDevBuf.data(),
                        scratch.acStateDevBuf.data(),
                        50);  // maximum number of fit iterations
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

namespace alpaka::trait {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE;
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit;

  //! The trait for getting the size of the block shared dynamic memory for Kernel_minimize.
  template <>
  struct BlockSharedMemDynSizeBytes<Kernel_minimize, Acc1D> {
    //! \return The size of the shared memory allocated for a block.
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_minimize const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      using ScalarType = ::ecal::multifit::SampleVector::Scalar;

      // return the amount of dynamic shared memory needed
      std::size_t bytes =
          2 * threadsPerBlock[0u] * elemsPerThread[0u] *
          calo::multifit::MapSymM<ScalarType, ::ecal::multifit::SampleVector::RowsAtCompileTime>::total *
          sizeof(ScalarType);
      return bytes;
    }
  };
}  // namespace alpaka::trait
