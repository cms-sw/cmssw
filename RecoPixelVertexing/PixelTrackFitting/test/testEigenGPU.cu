#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

#ifdef USE_BL
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"
#else
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#endif

#include "test_common.h"

using namespace Eigen;

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 5 * 1024; }
  constexpr uint32_t stride() { return maxNumberOfTracks(); }
  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()>>;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()>>;
  // fast fit
  using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()>>;

}  // namespace Rfit

template <int N>
__global__ void kernelPrintSizes(double* __restrict__ phits, float* __restrict__ phits_ge) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, 4);
  if (i != 0)
    return;
  printf("GPU sizes %lu %lu %lu %lu %lu\n",
         sizeof(hits[i]),
         sizeof(hits_ge[i]),
         sizeof(Vector4d),
         sizeof(Rfit::line_fit),
         sizeof(Rfit::circle_fit));
}

template <int N>
__global__ void kernelFastFit(double* __restrict__ phits, double* __restrict__ presults) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d result(presults + i, 4);
#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, result);
#else
  Rfit::Fast_fit(hits, result);
#endif
}

#ifdef USE_BL

template <int N>
__global__ void kernelBrokenLineFit(double* __restrict__ phits,
                                    float* __restrict__ phits_ge,
                                    double* __restrict__ pfast_fit_input,
                                    double B,
                                    Rfit::circle_fit* circle_fit,
                                    Rfit::line_fit* line_fit) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);

  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;

  auto& line_fit_results = line_fit[i];
  auto& circle_fit_results = circle_fit[i];

  BrokenLine::prepareBrokenLineData(hits, fast_fit_input, B, data);
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_input, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_input, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();

#ifdef TEST_DEBUG
  if (0 == i) {
    printf("Circle param %f,%f,%f\n", circle_fit[i].par(0), circle_fit[i].par(1), circle_fit[i].par(2));
  }
#endif
}

#else

template <int N>
__global__ void kernelCircleFit(double* __restrict__ phits,
                                float* __restrict__ phits_ge,
                                double* __restrict__ pfast_fit_input,
                                double B,
                                Rfit::circle_fit* circle_fit_resultsGPU) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);

  constexpr auto n = N;

  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, n).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov = MatrixXd::Zero(2 * n, 2 * n);
  Rfit::loadCovariance2D(hits_ge, hits_cov);

#ifdef TEST_DEBUG
  if (0 == i) {
    printf("hits %f, %f\n", hits.block(0, 0, 2, n)(0, 0), hits.block(0, 0, 2, n)(0, 1));
    printf("hits %f, %f\n", hits.block(0, 0, 2, n)(1, 0), hits.block(0, 0, 2, n)(1, 1));
    printf("fast_fit_input(0): %f\n", fast_fit_input(0));
    printf("fast_fit_input(1): %f\n", fast_fit_input(1));
    printf("fast_fit_input(2): %f\n", fast_fit_input(2));
    printf("fast_fit_input(3): %f\n", fast_fit_input(3));
    printf("rad(0,0): %f\n", rad(0, 0));
    printf("rad(1,1): %f\n", rad(1, 1));
    printf("rad(2,2): %f\n", rad(2, 2));
    printf("hits_cov(0,0): %f\n", (*hits_cov)(0, 0));
    printf("hits_cov(1,1): %f\n", (*hits_cov)(1, 1));
    printf("hits_cov(2,2): %f\n", (*hits_cov)(2, 2));
    printf("hits_cov(11,11): %f\n", (*hits_cov)(11, 11));
    printf("B: %f\n", B);
  }
#endif
  circle_fit_resultsGPU[i] = Rfit::Circle_fit(hits.block(0, 0, 2, n), hits_cov, fast_fit_input, rad, B, true);
#ifdef TEST_DEBUG
  if (0 == i) {
    printf("Circle param %f,%f,%f\n",
           circle_fit_resultsGPU[i].par(0),
           circle_fit_resultsGPU[i].par(1),
           circle_fit_resultsGPU[i].par(2));
  }
#endif
}

template <int N>
__global__ void kernelLineFit(double* __restrict__ phits,
                              float* __restrict__ phits_ge,
                              double B,
                              Rfit::circle_fit* circle_fit,
                              double* __restrict__ pfast_fit_input,
                              Rfit::line_fit* line_fit) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);
  line_fit[i] = Rfit::Line_fit(hits, hits_ge, circle_fit[i], fast_fit_input, B, true);
}
#endif

template <typename M3xN, typename M6xN>
__device__ __host__ void fillHitsAndHitsCov(M3xN& hits, M6xN& hits_ge) {
  constexpr uint32_t N = M3xN::ColsAtCompileTime;

  if (N == 5) {
    hits << 2.934787, 6.314229, 8.936963, 10.360559, 12.856387, 0.773211, 1.816356, 2.765734, 3.330824, 4.422212,
        -10.980247, -23.162731, -32.759060, -38.061260, -47.518867;
    hits_ge.col(0) << 1.424715e-07, -4.996975e-07, 1.752614e-06, 3.660689e-11, 1.644638e-09, 7.346080e-05;
    hits_ge.col(1) << 6.899177e-08, -1.873414e-07, 5.087101e-07, -2.078806e-10, -2.210498e-11, 4.346079e-06;
    hits_ge.col(2) << 1.406273e-06, 4.042467e-07, 6.391180e-07, -3.141497e-07, 6.513821e-08, 1.163863e-07;
    hits_ge.col(3) << 1.176358e-06, 2.154100e-07, 5.072816e-07, -8.161219e-08, 1.437878e-07, 5.951832e-08;
    hits_ge.col(4) << 2.852843e-05, 7.956492e-06, 3.117701e-06, -1.060541e-06, 8.777413e-09, 1.426417e-07;
    return;
  }

  if (N > 3)
    hits << 1.98645, 4.72598, 7.65632, 11.3151, 2.18002, 4.88864, 7.75845, 11.3134, 2.46338, 6.99838, 11.808, 17.793;
  else
    hits << 1.98645, 4.72598, 7.65632, 2.18002, 4.88864, 7.75845, 2.46338, 6.99838, 11.808;

  hits_ge.col(0)[0] = 7.14652e-06;
  hits_ge.col(1)[0] = 2.15789e-06;
  hits_ge.col(2)[0] = 1.63328e-06;
  if (N > 3)
    hits_ge.col(3)[0] = 6.27919e-06;
  hits_ge.col(0)[2] = 6.10348e-06;
  hits_ge.col(1)[2] = 2.08211e-06;
  hits_ge.col(2)[2] = 1.61672e-06;
  if (N > 3)
    hits_ge.col(3)[2] = 6.28081e-06;
  hits_ge.col(0)[5] = 5.184e-05;
  hits_ge.col(1)[5] = 1.444e-05;
  hits_ge.col(2)[5] = 6.25e-06;
  if (N > 3)
    hits_ge.col(3)[5] = 3.136e-05;
  hits_ge.col(0)[1] = -5.60077e-06;
  hits_ge.col(1)[1] = -1.11936e-06;
  hits_ge.col(2)[1] = -6.24945e-07;
  if (N > 3)
    hits_ge.col(3)[1] = -5.28e-06;
}

template <int N>
__global__ void kernelFillHitsAndHitsCov(double* __restrict__ phits, float* phits_ge) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);
  hits_ge = MatrixXf::Zero(6, N);
  fillHitsAndHitsCov(hits, hits_ge);
}

template <int N>
void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd<N> hits;
  Rfit::Matrix6xNf<N> hits_ge = MatrixXf::Zero(6, N);
  double* hitsGPU = nullptr;
  ;
  float* hits_geGPU = nullptr;
  double* fast_fit_resultsGPU = nullptr;
  double* fast_fit_resultsGPUret = new double[Rfit::maxNumberOfTracks() * sizeof(Vector4d)];
  Rfit::circle_fit* circle_fit_resultsGPU = nullptr;
  Rfit::circle_fit* circle_fit_resultsGPUret = new Rfit::circle_fit();
  Rfit::line_fit* line_fit_resultsGPU = nullptr;
  Rfit::line_fit* line_fit_resultsGPUret = new Rfit::line_fit();

  fillHitsAndHitsCov(hits, hits_ge);

  std::cout << "sizes " << N << ' ' << sizeof(hits) << ' ' << sizeof(hits_ge) << ' ' << sizeof(Vector4d) << ' '
            << sizeof(Rfit::line_fit) << ' ' << sizeof(Rfit::circle_fit) << std::endl;

  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

  // FAST_FIT_CPU
#ifdef USE_BL
  Vector4d fast_fit_results;
  BrokenLine::BL_Fast_fit(hits, fast_fit_results);
#else
  Vector4d fast_fit_results;
  Rfit::Fast_fit(hits, fast_fit_results);
#endif
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;

  // for timing    purposes we fit    4096 tracks
  constexpr uint32_t Ntracks = 4096;
  cudaCheck(cudaMalloc(&hitsGPU, Rfit::maxNumberOfTracks() * sizeof(Rfit::Matrix3xNd<N>)));
  cudaCheck(cudaMalloc(&hits_geGPU, Rfit::maxNumberOfTracks() * sizeof(Rfit::Matrix6xNf<N>)));
  cudaCheck(cudaMalloc(&fast_fit_resultsGPU, Rfit::maxNumberOfTracks() * sizeof(Vector4d)));
  cudaCheck(cudaMalloc(&line_fit_resultsGPU, Rfit::maxNumberOfTracks() * sizeof(Rfit::line_fit)));
  cudaCheck(cudaMalloc(&circle_fit_resultsGPU, Rfit::maxNumberOfTracks() * sizeof(Rfit::circle_fit)));

  cudaCheck(cudaMemset(fast_fit_resultsGPU, 0, Rfit::maxNumberOfTracks() * sizeof(Vector4d)));
  cudaCheck(cudaMemset(line_fit_resultsGPU, 0, Rfit::maxNumberOfTracks() * sizeof(Rfit::line_fit)));

  kernelPrintSizes<N><<<Ntracks / 64, 64>>>(hitsGPU, hits_geGPU);
  kernelFillHitsAndHitsCov<N><<<Ntracks / 64, 64>>>(hitsGPU, hits_geGPU);

  // FAST_FIT GPU
  kernelFastFit<N><<<Ntracks / 64, 64>>>(hitsGPU, fast_fit_resultsGPU);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(fast_fit_resultsGPUret,
                       fast_fit_resultsGPU,
                       Rfit::maxNumberOfTracks() * sizeof(Vector4d),
                       cudaMemcpyDeviceToHost));
  Rfit::Map4d fast_fit(fast_fit_resultsGPUret + 10, 4);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]): GPU\n" << fast_fit << std::endl;
  assert(isEqualFuzzy(fast_fit_results, fast_fit));

#ifdef USE_BL
  // CIRCLE AND LINE FIT CPU
  BrokenLine::PreparedBrokenLineData<N> data;
  BrokenLine::karimaki_circle_fit circle_fit_results;
  Rfit::line_fit line_fit_results;
  Rfit::Matrix3d Jacob;
  BrokenLine::prepareBrokenLineData(hits, fast_fit_results, B, data);
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_results, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_results, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();

  // fit on GPU
  kernelBrokenLineFit<N>
      <<<Ntracks / 64, 64>>>(hitsGPU, hits_geGPU, fast_fit_resultsGPU, B, circle_fit_resultsGPU, line_fit_resultsGPU);
  cudaDeviceSynchronize();

#else
  // CIRCLE_FIT CPU
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge, hits_cov);
  Rfit::circle_fit circle_fit_results =
      Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit_results, rad, B, true);

  // CIRCLE_FIT GPU
  kernelCircleFit<N><<<Ntracks / 64, 64>>>(hitsGPU, hits_geGPU, fast_fit_resultsGPU, B, circle_fit_resultsGPU);
  cudaDeviceSynchronize();

  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);

  kernelLineFit<N>
      <<<Ntracks / 64, 64>>>(hitsGPU, hits_geGPU, B, circle_fit_resultsGPU, fast_fit_resultsGPU, line_fit_resultsGPU);
  cudaDeviceSynchronize();
#endif

  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;

  cudaCheck(
      cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU, sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost));
  std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << std::endl;
  assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par));

  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;
  // LINE_FIT GPU
  cudaCheck(cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost));
  std::cout << "Fitted values (LineFit) GPU:\n" << line_fit_resultsGPUret->par << std::endl;
  assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par, N == 5 ? 1e-4 : 1e-6));  // requires fma on CPU

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
  std::cout << "Fitted cov (CircleFit) GPU:\n" << circle_fit_resultsGPUret->cov << std::endl;
  std::cout << "Fitted cov (LineFit): GPU\n" << line_fit_resultsGPUret->cov << std::endl;
}

int main(int argc, char* argv[]) {
  cms::cudatest::requireDevices();

  testFit<4>();
  testFit<3>();
  testFit<5>();

  std::cout << "TEST FIT, NO ERRORS" << std::endl;

  return 0;
}
