#ifndef RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h
#define RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h

#include "DataFormats/Math/interface/EigenComputations.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

namespace ecal {
  namespace multifit {

    // TODO: add active bxs
    template <typename MatrixType, typename VectorType>
    __device__ void fnnls(MatrixType const& AtA,
                          VectorType const& Atb,
                          VectorType& solution,
                          int& npassive,
                          calo::multifit::ColumnVector<VectorType::RowsAtCompileTime, int>& pulseOffsets,
                          calo::multifit::MapSymM<float, VectorType::RowsAtCompileTime>& matrixL,
                          double const eps,
                          int const maxIterations) {
      // constants
      constexpr auto NPULSES = VectorType::RowsAtCompileTime;

      // to keep track of where to terminate if converged
      Eigen::Index w_max_idx_prev = 0;
      float w_max_prev = 0;
      auto eps_to_use = eps;
      bool recompute = false;

      // used throughout
      VectorType s;
      float reg_b[NPULSES];
      //float matrixLStorage[MapSymM<float, NPULSES>::total];
      //MapSymM<float, NPULSES> matrixL{matrixLStorage};

      int iter = 0;
      while (true) {
        if (iter > 0 || npassive == 0) {
          auto const nactive = NPULSES - npassive;
          // exit if there are no more pulses to constrain
          if (nactive == 0)
            break;

          // compute the gradient
          //w.tail(nactive) = Atb.tail(nactive) - (AtA * solution).tail(nactive);
          Eigen::Index w_max_idx;
          float w_max = -std::numeric_limits<float>::max();
          for (int icol = npassive; icol < NPULSES; icol++) {
            auto const icol_real = pulseOffsets(icol);
            auto const atb = Atb(icol_real);
            float sum = 0;
#pragma unroll
            for (int counter = 0; counter < NPULSES; counter++)
              sum += counter > icol_real ? AtA(counter, icol_real) * solution(counter)
                                         : AtA(icol_real, counter) * solution(counter);

            auto const w = atb - sum;
            if (w > w_max) {
              w_max = w;
              w_max_idx = icol - npassive;
            }
          }

          // check for convergence
          if (w_max < eps_to_use || w_max_idx == w_max_idx_prev && w_max == w_max_prev)
            break;

          if (iter >= maxIterations)
            break;

          w_max_prev = w_max;
          w_max_idx_prev = w_max_idx;

          // move index to the right part of the vector
          w_max_idx += npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(w_max_idx));
          ++npassive;
        }

        // inner loop
        while (true) {
          if (npassive == 0)
            break;

          //s.head(npassive)
          //auto const& matrixL =
          //    AtA.topLeftCorner(npassive, npassive)
          //        .llt().matrixL();
          //.solve(Atb.head(npassive));
          if (recompute || iter == 0)
            compute_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);
          else
            update_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);

          // run backward substituion
          s(npassive - 1) = reg_b[npassive - 1] / matrixL(npassive - 1, npassive - 1);
          for (int i = npassive - 2; i >= 0; --i) {
            float total = 0;
            for (int j = i + 1; j < npassive; j++)
              total += matrixL(j, i) * s(j);

            s(i) = (reg_b[i] - total) / matrixL(i, i);
          }

          // done if solution values are all positive
          bool hasNegative = false;
          bool hasNans = false;
          for (int counter = 0; counter < npassive; counter++) {
            auto const s_ii = s(counter);
            hasNegative |= s_ii <= 0;
            hasNans |= std::isnan(s_ii);
          }

          // FIXME: temporary solution. my cholesky impl is unstable yielding nans
          // this check removes nans - do not accept solution unless all values
          // are stable
          if (hasNans)
            break;
          if (!hasNegative) {
            for (int i = 0; i < npassive; i++) {
              auto const i_real = pulseOffsets(i);
              solution(i_real) = s(i);
            }
            //solution.head(npassive) = s.head(npassive);
            recompute = false;
            break;
          }

          // there were negative values -> have to recompute the whole decomp
          recompute = true;

          auto alpha = std::numeric_limits<float>::max();
          Eigen::Index alpha_idx = 0, alpha_idx_real = 0;
          for (int i = 0; i < npassive; i++) {
            if (s[i] <= 0.) {
              auto const i_real = pulseOffsets(i);
              auto const ratio = solution[i_real] / (solution[i_real] - s[i]);
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
                alpha_idx_real = i_real;
              }
            }
          }

          // upadte solution
          for (int i = 0; i < npassive; i++) {
            auto const i_real = pulseOffsets(i);
            solution(i_real) += alpha * (s(i) - solution(i_real));
          }
          //solution.head(npassive) += alpha *
          //    (s.head(npassive) - solution.head(npassive));
          solution[alpha_idx_real] = 0;
          --npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(alpha_idx));
        }

        // as in cpu
        ++iter;
        if (iter % 16 == 0)
          eps_to_use *= 2;
      }
    }

  }  // namespace multifit
}  // namespace ecal

namespace ecal {
  namespace reconstruction {

    __device__ uint32_t hashedIndexEB(uint32_t id);

    __device__ uint32_t hashedIndexEE(uint32_t id);

    __device__ int laser_monitoring_region_EB(uint32_t id);

    __device__ int laser_monitoring_region_EE(uint32_t id);

  }  // namespace reconstruction
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h
