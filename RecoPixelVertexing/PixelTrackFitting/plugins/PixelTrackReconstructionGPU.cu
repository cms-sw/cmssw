#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "PixelTrackReconstructionGPU.h"

using namespace Eigen;

__global__ void
KernelFastFitAllHits(float *hits_and_covariances,
    int hits_in_fit,
    int cumulative_size,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd<4> *hits,
    Eigen::Matrix<float,6,4> *hits_ge,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  // Reshape Eigen components from hits_and_covariances, using proper thread and block indices
  // Perform the fit
  // Store the results in the proper vector, using again correct indices

  // Loop for hits_in_fit times:
  //   first 3 are the points
  //   the rest is the covariance matrix, 3x3
  int start = (blockIdx.x * blockDim.x + threadIdx.x) * hits_in_fit * 12;
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (start >= cumulative_size) {
    return;
  }

#ifdef GPU_DEBUG
  printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, start: %d, cumulative_size: %d\n",
      blockDim.x, blockIdx.x, threadIdx.x, start, cumulative_size);
#endif


  // Prepare data structure (stack)
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    hits[helix_start].col(i) << hits_and_covariances[start],
        hits_and_covariances[start + 1], hits_and_covariances[start + 2];
    start += 3;

    hits_ge[helix_start].col(i) << hits_and_covariances[start],
        hits_and_covariances[start + 1], hits_and_covariances[start + 2],
        hits_and_covariances[start + 3], hits_and_covariances[start + 4],
        hits_and_covariances[start + 5];
    start += 6;
  }

  Rfit::Fast_fit(hits[helix_start],fast_fit[helix_start]);
}

__global__ void
KernelCircleFitAllHits(float *hits_and_covariances, int hits_in_fit,
                       int cumulative_size, float B, Rfit::helix_fit *results,
                       Rfit::Matrix3xNd<4> *hits, Eigen::Matrix<float,6,4> *hits_ge,
                       Rfit::circle_fit *circle_fit, Vector4d *fast_fit,
                       Rfit::line_fit *line_fit)
{
  // Reshape Eigen components from hits_and_covariances, using proper thread and block indices
  // Perform the fit
  // Store the results in the proper vector, using again correct indices

  // Loop for hits_in_fit times:
  //   first 3 are the points
  //   the rest is the covariance matrix, 3x3
  int start = (blockIdx.x * blockDim.x + threadIdx.x) * hits_in_fit * 12;
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (start >= cumulative_size) {
    return;
  }

#ifdef GPU_DEBUG
    printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, start: %d, "
           "cumulative_size: %d\n",
           blockDim.x, blockIdx.x, threadIdx.x, start, cumulative_size);
#endif
  u_int n = hits[helix_start].cols();

  constexpr uint32_t N = 4;

  Rfit::VectorNd<N> rad = (hits[helix_start].block(0, 0, 2, n).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov =  MatrixXd::Zero(2 * n, 2 * n);
  Rfit::loadCovariance2D(hits_ge[helix_start],hits_cov);
  circle_fit[helix_start] =
      Rfit::Circle_fit(hits[helix_start].block(0, 0, 2, n),
                       hits_cov,
                       fast_fit[helix_start], rad, B, true);

#ifdef GPU_DEBUG
    printf("KernelCircleFitAllHits circle.par(0): %d %f\n", helix_start,
           circle_fit[helix_start].par(0));
    printf("KernelCircleFitAllHits circle.par(1): %d %f\n", helix_start,
           circle_fit[helix_start].par(1));
    printf("KernelCircleFitAllHits circle.par(2): %d %f\n", helix_start,
           circle_fit[helix_start].par(2));
#endif

}

__global__ void
KernelLineFitAllHits(float *hits_and_covariances, int hits_in_fit,
                     int cumulative_size, float B, Rfit::helix_fit *results,
                      Rfit::Matrix3xNd<4> *hits, Eigen::Matrix<float,6,4> *hits_ge,
                     Rfit::circle_fit *circle_fit, Vector4d *fast_fit,
                     Rfit::line_fit *line_fit)
{
  // Reshape Eigen components from hits_and_covariances, using proper thread and block indices
  // Perform the fit
  // Store the results in the proper vector, using again correct indices

  // Loop for hits_in_fit times:
  //   first 3 are the points
  //   the rest is the covariance matrix, 3x3
  int start = (blockIdx.x * blockDim.x + threadIdx.x) * hits_in_fit * 12;
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (start >= cumulative_size) {
    return;
  }

#ifdef GPU_DEBUG

    printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, start: %d, "
           "cumulative_size: %d\n",
           blockDim.x, blockIdx.x, threadIdx.x, start, cumulative_size);
#endif

  line_fit[helix_start] =
      Rfit::Line_fit(hits[helix_start], hits_ge[helix_start],
                     circle_fit[helix_start], fast_fit[helix_start], B, true);

  par_uvrtopak(circle_fit[helix_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  Rfit::helix_fit &helix = results[helix_start];
  helix.par << circle_fit[helix_start].par, line_fit[helix_start].par;

  // TODO: pass properly error booleans

  helix.cov = MatrixXd::Zero(5, 5);
  helix.cov.block(0, 0, 3, 3) = circle_fit[helix_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit[helix_start].cov;

  helix.q = circle_fit[helix_start].q;
  helix.chi2_circle = circle_fit[helix_start].chi2;
  helix.chi2_line = line_fit[helix_start].chi2;

#ifdef GPU_DEBUG

    printf("KernelLineFitAllHits line.par(0): %d %f\n", helix_start,
           circle_fit[helix_start].par(0));
    printf("KernelLineFitAllHits line.par(1): %d %f\n", helix_start,
           line_fit[helix_start].par(1));
#endif
}

void PixelTrackReconstructionGPU::launchKernelFit(
    float *hits_and_covariancesGPU, int cumulative_size, int hits_in_fit,
    float B, Rfit::helix_fit *results)
{
  const dim3 threads_per_block(32, 1);
  int num_blocks = cumulative_size / (hits_in_fit * 12) / threads_per_block.x + 1;
  auto numberOfSeeds = cumulative_size / (hits_in_fit * 12);

  Rfit::Matrix3xNd<4> *hitsGPU;
  cudaCheck(cudaMalloc(&hitsGPU, 48 * numberOfSeeds * sizeof(Rfit::Matrix3xNd<4>)));
  cudaCheck(cudaMemset(hitsGPU, 0x00, 48 * numberOfSeeds * sizeof(Rfit::Matrix3xNd<4>)));

  Eigen::Matrix<float,6,4> *hits_geGPU = nullptr;
  cudaCheck(cudaMalloc(&hits_geGPU, 48 * numberOfSeeds * sizeof(Eigen::Matrix<float,6,4>)));
  cudaCheck(cudaMemset(hits_geGPU, 0x00, 48 * numberOfSeeds * sizeof(Eigen::Matrix<float,6,4>)));

  Vector4d *fast_fit_resultsGPU = nullptr;
  cudaCheck(cudaMalloc(&fast_fit_resultsGPU, 48 * numberOfSeeds * sizeof(Vector4d)));
  cudaCheck(cudaMemset(fast_fit_resultsGPU, 0x00, 48 * numberOfSeeds * sizeof(Vector4d)));

  Rfit::circle_fit *circle_fit_resultsGPU = nullptr;
  cudaCheck(cudaMalloc(&circle_fit_resultsGPU, 48 * numberOfSeeds * sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemset(circle_fit_resultsGPU, 0x00, 48 * numberOfSeeds * sizeof(Rfit::circle_fit)));

  Rfit::line_fit *line_fit_resultsGPU = nullptr;
  cudaCheck(cudaMalloc(&line_fit_resultsGPU, numberOfSeeds * sizeof(Rfit::line_fit)));
  cudaCheck(cudaMemset(line_fit_resultsGPU, 0x00, numberOfSeeds * sizeof(Rfit::line_fit)));

  KernelFastFitAllHits<<<num_blocks, threads_per_block>>>(
      hits_and_covariancesGPU, hits_in_fit, cumulative_size, B, results,
      hitsGPU, hits_geGPU, circle_fit_resultsGPU, fast_fit_resultsGPU,
      line_fit_resultsGPU);
  cudaCheck(cudaGetLastError());

  KernelCircleFitAllHits<<<num_blocks, threads_per_block>>>(
      hits_and_covariancesGPU, hits_in_fit, cumulative_size, B, results,
      hitsGPU, hits_geGPU, circle_fit_resultsGPU, fast_fit_resultsGPU,
      line_fit_resultsGPU);
  cudaCheck(cudaGetLastError());

  KernelLineFitAllHits<<<num_blocks, threads_per_block>>>(
      hits_and_covariancesGPU, hits_in_fit, cumulative_size, B, results,
      hitsGPU, hits_geGPU, circle_fit_resultsGPU, fast_fit_resultsGPU,
      line_fit_resultsGPU);
  cudaCheck(cudaGetLastError());

  cudaFree(hitsGPU);
  cudaFree(hits_geGPU);
  cudaFree(fast_fit_resultsGPU);
  cudaFree(circle_fit_resultsGPU);
  cudaFree(line_fit_resultsGPU);
}
