#ifndef DataFormats_CaloRecHit_interface_MultifitComputations_h
#define DataFormats_CaloRecHit_interface_MultifitComputations_h

#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

namespace calo {
  namespace multifit {

    template <int NROWS, int NCOLS>
    using ColMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::ColMajor>;

    template <int NROWS, int NCOLS>
    using RowMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::RowMajor>;

    template <int SIZE, typename T = float>
    using ColumnVector = Eigen::Matrix<T, SIZE, 1>;

    template <int SIZE, typename T = float>
    using RowVector = Eigen::Matrix<T, 1, SIZE>;

    // FIXME: provide specialization for Row Major layout
    template <typename T, int Stride, int Order = Eigen::ColMajor>
    struct MapSymM {
      using type = T;
      using base_type = typename std::remove_const<type>::type;

      static constexpr int total = Stride * (Stride + 1) / 2;
      static constexpr int stride = Stride;
      T* data;

      EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC MapSymM(T* data) : data{data} {}

      EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC T const& operator()(int const row, int const col) const {
        auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
        auto const index = total - tmp + row - col;
        return data[index];
      }

      template <typename U = T>
      EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC typename std::enable_if<std::is_same<base_type, U>::value, base_type>::type&
      operator()(int const row, int const col) {
        auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
        auto const index = total - tmp + row - col;
        return data[index];
      }
    };

    // FIXME: either use/modify/improve eigen or make this more generic
    // this is a map for a pulse matrix to building a 2d matrix for each channel
    // and hide indexing
    template <typename T>
    struct MapMForPM {
      using type = T;
      using base_type = typename std::remove_cv<type>::type;

      type* data;
      EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC MapMForPM(type* data) : data{data} {}

      EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC base_type operator()(int const row, int const col) const {
        auto const index = 2 - col + row;
        return index >= 0 ? data[index] : 0;
      }
    };

    // simple/trivial cholesky decomposition impl
    template <typename MatrixType1, typename MatrixType2>
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void compute_decomposition_unrolled(MatrixType1& L, MatrixType2 const& M) {
      auto const sqrtm_0_0 = std::sqrt(M(0, 0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;

#pragma unroll
      for (int i = 1; i < MatrixType1::stride; i++) {
        T sumsq{0};
        for (int j = 0; j < i; j++) {
          T sumsq2{0};
          auto const m_i_j = M(i, j);
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
        }

        auto const l_i_i = std::sqrt(M(i, i) - sumsq);
        L(i, i) = l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2>
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void compute_decomposition(MatrixType1& L,
                                                                     MatrixType2 const& M,
                                                                     int const N) {
      auto const sqrtm_0_0 = std::sqrt(M(0, 0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;

      for (int i = 1; i < N; i++) {
        T sumsq{0};
        for (int j = 0; j < i; j++) {
          T sumsq2{0};
          auto const m_i_j = M(i, j);
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
        }

        auto const l_i_i = std::sqrt(M(i, i) - sumsq);
        L(i, i) = l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2, typename VectorType>
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void compute_decomposition_forwardsubst_with_offsets(
        MatrixType1& L,
        MatrixType2 const& M,
        float b[MatrixType1::stride],
        VectorType const& Atb,
        int const N,
        ColumnVector<MatrixType1::stride, int> const& pulseOffsets) {
      auto const real_0 = pulseOffsets(0);
      auto const sqrtm_0_0 = std::sqrt(M(real_0, real_0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;
      b[0] = Atb(real_0) / sqrtm_0_0;

      for (int i = 1; i < N; i++) {
        auto const i_real = pulseOffsets(i);
        T sumsq{0};
        T total = 0;
        auto const atb = Atb(i_real);
        for (int j = 0; j < i; j++) {
          auto const j_real = pulseOffsets(j);
          T sumsq2{0};
          auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
          total += value_i_j * b[j];
        }

        auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
        L(i, i) = l_i_i;
        b[i] = (atb - total) / l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2, typename VectorType>
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void update_decomposition_forwardsubst_with_offsets(
        MatrixType1& L,
        MatrixType2 const& M,
        float b[MatrixType1::stride],
        VectorType const& Atb,
        int const N,
        ColumnVector<MatrixType1::stride, int> const& pulseOffsets) {
      using T = typename MatrixType1::base_type;
      auto const i = N - 1;
      auto const i_real = pulseOffsets(i);
      T sumsq{0};
      T total = 0;
      for (int j = 0; j < i; j++) {
        auto const j_real = pulseOffsets(j);
        T sumsq2{0};
        auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));
        for (int k = 0; k < j; ++k)
          sumsq2 += L(i, k) * L(j, k);

        auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
        L(i, j) = value_i_j;
        sumsq += value_i_j * value_i_j;

        total += value_i_j * b[j];
      }

      auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
      L(i, i) = l_i_i;
      b[i] = (Atb(i_real) - total) / l_i_i;
    }

    template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
    EIGEN_DEVICE_FUNC void solve_forward_subst_matrix(MatrixType1& A,
                                                      MatrixType2 const& pulseMatrixView,
                                                      MatrixType3 const& matrixL) {
      // FIXME: this assumes pulses are on columns and samples on rows
      constexpr auto NPULSES = MatrixType2::ColsAtCompileTime;
      constexpr auto NSAMPLES = MatrixType2::RowsAtCompileTime;

#pragma unroll
      for (int icol = 0; icol < NPULSES; icol++) {
        float reg_b[NSAMPLES];
        float reg_L[NSAMPLES];

// preload a column and load column 0 of cholesky
#pragma unroll
        for (int i = 0; i < NSAMPLES; i++) {
#ifdef __CUDA_ARCH__
          // load through the read-only cache
          reg_b[i] = __ldg(&pulseMatrixView.coeffRef(i, icol));
#else
          reg_b[i] = pulseMatrixView.coeffRef(i, icol);
#endif  // __CUDA_ARCH__
          reg_L[i] = matrixL(i, 0);
        }

        // compute x0 and store it
        auto x_prev = reg_b[0] / reg_L[0];
        A(0, icol) = x_prev;

// iterate
#pragma unroll
        for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
          for (int counter = iL; counter < NSAMPLES; counter++)
            reg_b[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
          for (int counter = iL; counter < NSAMPLES; counter++)
            reg_L[counter] = matrixL(counter, iL);

          // compute the next x for M(iL, icol)
          x_prev = reg_b[iL] / reg_L[iL];

          // store the result value
          A(iL, icol) = x_prev;
        }
      }
    }

    template <typename MatrixType1, typename MatrixType2>
    EIGEN_DEVICE_FUNC void solve_forward_subst_vector(float reg_b[MatrixType1::RowsAtCompileTime],
                                                      MatrixType1 inputAmplitudesView,
                                                      MatrixType2 matrixL) {
      constexpr auto NSAMPLES = MatrixType1::RowsAtCompileTime;

      float reg_b_tmp[NSAMPLES];
      float reg_L[NSAMPLES];

// preload a column and load column 0 of cholesky
#pragma unroll
      for (int i = 0; i < NSAMPLES; i++) {
        reg_b_tmp[i] = inputAmplitudesView(i);
        reg_L[i] = matrixL(i, 0);
      }

      // compute x0 and store it
      auto x_prev = reg_b_tmp[0] / reg_L[0];
      reg_b[0] = x_prev;

// iterate
#pragma unroll
      for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
        for (int counter = iL; counter < NSAMPLES; counter++)
          reg_b_tmp[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
        for (int counter = iL; counter < NSAMPLES; counter++)
          reg_L[counter] = matrixL(counter, iL);

        // compute the next x for M(iL, icol)
        x_prev = reg_b_tmp[iL] / reg_L[iL];

        // store the result value
        reg_b[iL] = x_prev;
      }
    }

    template <typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4>
    EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void calculateChiSq(MatrixType1 const& matrixL,
                                                              MatrixType2 const& pulseMatrixView,
                                                              MatrixType3 const& resultAmplitudesVector,
                                                              MatrixType4 const& inputAmplitudesView,
                                                              float& chi2) {
      // FIXME: this assumes pulses are on columns and samples on rows
      constexpr auto NPULSES = MatrixType2::ColsAtCompileTime;
      constexpr auto NSAMPLES = MatrixType2::RowsAtCompileTime;

      // replace pulseMatrixView * resultAmplitudesVector - inputAmplitudesView
      // NOTE:
      float accum[NSAMPLES];
      {
        float results[NPULSES];

        // preload results and permute according to the pulse offsets /////////////// ??? this is not done in ECAL
#pragma unroll
        for (int counter = 0; counter < NPULSES; counter++) {
          results[counter] = resultAmplitudesVector[counter];
        }

        // load accum
#pragma unroll
        for (int counter = 0; counter < NSAMPLES; counter++)
          accum[counter] = -inputAmplitudesView(counter);

        // iterate
        for (int icol = 0; icol < NPULSES; icol++) {
          float pm_col[NSAMPLES];

          // preload a column of pulse matrix
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
#ifdef __CUDA_ARCH__
            pm_col[counter] = __ldg(&pulseMatrixView.coeffRef(counter, icol));
#else
            pm_col[counter] = pulseMatrixView.coeffRef(counter, icol);
#endif

            // accum
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            accum[counter] += results[icol] * pm_col[counter];
        }
      }

      // compute chi2 and check that there is no rotation
      // chi2 = matrixDecomposition
      //    .matrixL()
      //    . solve(mapAccum)
      //            .solve(pulseMatrixView * resultAmplitudesVector - inputAmplitudesView)
      //    .squaredNorm();

      {
        float reg_L[NSAMPLES];
        float accumSum = 0;

        // preload a column and load column 0 of cholesky
#pragma unroll
        for (int i = 0; i < NSAMPLES; i++) {
          reg_L[i] = matrixL(i, 0);
        }

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

          // store the result value
          accumSum += x_prev * x_prev;
        }

        chi2 = accumSum;
      }
    }

    // TODO: add active bxs
    template <typename MatrixType, typename VectorType>
    EIGEN_DEVICE_FUNC void fnnls(MatrixType const& AtA,
                                 VectorType const& Atb,
                                 VectorType& solution,
                                 int& npassive,
                                 ColumnVector<VectorType::RowsAtCompileTime, int>& pulseOffsets,
                                 MapSymM<float, VectorType::RowsAtCompileTime>& matrixL,
                                 double eps,                    // convergence condition
                                 const int maxIterations,       // maximum number of iterations
                                 const int relaxationPeriod,    // every "relaxationPeriod" iterations
                                 const int relaxationFactor) {  // multiply "eps" by "relaxationFactor"
      // constants
      constexpr auto NPULSES = VectorType::RowsAtCompileTime;

      // to keep track of where to terminate if converged
      Eigen::Index w_max_idx_prev = 0;
      float w_max_prev = 0;
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
          if (w_max < eps || (w_max_idx == w_max_idx_prev && w_max == w_max_prev))
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
        if (iter % relaxationPeriod == 0)
          eps *= relaxationFactor;
      }
    }

  }  // namespace multifit
}  // namespace calo

#endif  // DataFormats_CaloRecHit_interface_MultifitComputations_h
