#include "CUDADataFormats/Track/interface/TrajectoryStateSoA.h"

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

__host__ __device__ Matrix5d loadCov(Vector5d const& e) {
  Matrix5d cov;
  for (int i = 0; i < 5; ++i)
    cov(i, i) = e(i) * e(i);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3 * std::sqrt(cov(i, i) * cov(j, j));  // this makes the matrix pos defined
      cov(i, j) = (i + j) % 2 ? -0.4 * v : 0.1 * v;
      cov(j, i) = cov(i, j);
    }
  }
  return cov;
}

using TS = TrajectoryStateSoA<128>;

__global__ void testTSSoA(TS* pts, int n) {
  assert(n <= 128);

  Vector5d par0;
  par0 << 0.2, 0.1, 3.5, 0.8, 0.1;
  Vector5d e0;
  e0 << 0.01, 0.01, 0.035, -0.03, -0.01;
  auto cov0 = loadCov(e0);

  TS& ts = *pts;

  int first = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = first; i < n; i += blockDim.x * gridDim.x) {
    ts.copyFromDense(par0, cov0, i);
    Vector5d par1;
    Matrix5d cov1;
    ts.copyToDense(par1, cov1, i);
    Vector5d delV = par1 - par0;
    Matrix5d delM = cov1 - cov0;
    for (int j = 0; j < 5; ++j) {
      assert(std::abs(delV(j)) < 1.e-5);
      for (auto k = j; k < 5; ++k) {
        assert(cov0(k, j) == cov0(j, k));
        assert(cov1(k, j) == cov1(j, k));
        assert(std::abs(delM(k, j)) < 1.e-5);
      }
    }
  }
}

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#endif

int main() {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();
#endif

  TS ts;

#ifdef __CUDACC__
  TS* ts_d;
  cudaCheck(cudaMalloc(&ts_d, sizeof(TS)));
  testTSSoA<<<1, 64>>>(ts_d, 128);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(&ts, ts_d, sizeof(TS), cudaMemcpyDefault));
  cudaCheck(cudaDeviceSynchronize());
#else
  testTSSoA(&ts, 128);
#endif
}
