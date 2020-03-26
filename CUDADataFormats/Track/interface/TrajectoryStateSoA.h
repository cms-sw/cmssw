#ifndef CUDADataFormatsTrackTrajectoryStateSOA_H
#define CUDADataFormatsTrackTrajectoryStateSOA_H

#include <Eigen/Dense>
#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"

template <int32_t S>
struct TrajectoryStateSoA {
  using Vector5f = Eigen::Matrix<float, 5, 1>;
  using Vector15f = Eigen::Matrix<float, 15, 1>;

  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Matrix5d = Eigen::Matrix<double, 5, 5>;

  static constexpr int32_t stride() { return S; }

  eigenSoA::MatrixSoA<Vector5f, S> state;
  eigenSoA::MatrixSoA<Vector15f, S> covariance;

  template <typename V3, typename M3, typename V2, typename M2>
  __host__ __device__ inline void copyFromCircle(
      V3 const& cp, M3 const& ccov, V2 const& lp, M2 const& lcov, float b, int32_t i) {
    state(i) << cp.template cast<float>(), lp.template cast<float>();
    state(i)(2) *= b;
    auto cov = covariance(i);
    cov(0) = ccov(0, 0);
    cov(1) = ccov(0, 1);
    cov(2) = b * float(ccov(0, 2));
    cov(4) = cov(3) = 0;
    cov(5) = ccov(1, 1);
    cov(6) = b * float(ccov(1, 2));
    cov(8) = cov(7) = 0;
    cov(9) = b * b * float(ccov(2, 2));
    cov(11) = cov(10) = 0;
    cov(12) = lcov(0, 0);
    cov(13) = lcov(0, 1);
    cov(14) = lcov(1, 1);
  }

  template <typename V5, typename M5>
  __host__ __device__ inline void copyFromDense(V5 const& v, M5 const& cov, int32_t i) {
    state(i) = v.template cast<float>();
    for (int j = 0, ind = 0; j < 5; ++j)
      for (auto k = j; k < 5; ++k)
        covariance(i)(ind++) = cov(j, k);
  }

  template <typename V5, typename M5>
  __host__ __device__ inline void copyToDense(V5& v, M5& cov, int32_t i) const {
    v = state(i).template cast<typename V5::Scalar>();
    for (int j = 0, ind = 0; j < 5; ++j) {
      cov(j, j) = covariance(i)(ind++);
      for (auto k = j + 1; k < 5; ++k)
        cov(k, j) = cov(j, k) = covariance(i)(ind++);
    }
  }
};

#endif  // CUDADataFormatsTrackTrajectoryStateSOA_H
