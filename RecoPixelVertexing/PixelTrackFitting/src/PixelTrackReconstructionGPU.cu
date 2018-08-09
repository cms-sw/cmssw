#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstructionGPU.h"

#ifndef GPU_DEBUG
#define GPU_DEBUG 0
#endif // GPU_DEBUG

using namespace Eigen;

__global__
void KernelFullFitAllHits(float * hits_and_covariances,
    int hits_in_fit,
    int cumulative_size,
    double B,
    Rfit::helix_fit * results) {
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

#if GPU_DEBUG
  printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, start: %d, cumulative_size: %d\n",
      blockDim.x, blockIdx.x, threadIdx.x, start, cumulative_size);
#endif

  Rfit::Matrix3xNd hits(3,hits_in_fit);
  Rfit::Matrix3Nd hits_cov(3 * hits_in_fit, 3 * hits_in_fit);

  // Prepare data structure (stack)
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    hits.col(i) << hits_and_covariances[start],
      hits_and_covariances[start+1],hits_and_covariances[start+2];
    start += 3;

    for (auto j = 0; j < 3; ++j) {
      for (auto l = 0; l < 3; ++l) {
        hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = hits_and_covariances[start];
        start++;
      }
    }
  }

#if GPU_DEBUG
  printf("KernelFullFitAllHits hits(0,0): %d\t%f\n", helix_start, hits(0,0));
  printf("KernelFullFitAllHits hits(0,1): %d\t%f\n", helix_start, hits(0,1));
  printf("KernelFullFitAllHits hits(0,2): %d\t%f\n", helix_start, hits(0,2));
  printf("KernelFullFitAllHits hits(0,3): %d\t%f\n", helix_start, hits(0,3));
  printf("KernelFullFitAllHits hits(1,0): %d\t%f\n", helix_start, hits(1,0));
  printf("KernelFullFitAllHits hits(1,1): %d\t%f\n", helix_start, hits(1,1));
  printf("KernelFullFitAllHits hits(1,2): %d\t%f\n", helix_start, hits(1,2));
  printf("KernelFullFitAllHits hits(1,3): %d\t%f\n", helix_start, hits(1,3));
  printf("KernelFullFitAllHits hits(2,0): %d\t%f\n", helix_start, hits(2,0));
  printf("KernelFullFitAllHits hits(2,1): %d\t%f\n", helix_start, hits(2,1));
  printf("KernelFullFitAllHits hits(2,2): %d\t%f\n", helix_start, hits(2,2));
  printf("KernelFullFitAllHits hits(2,3): %d\t%f\n", helix_start, hits(2,3));
  Rfit::printIt(&hits);
  Rfit::printIt(&hits_cov);
#endif

  // Perform actual fit
  Vector4d fast_fit = Rfit::Fast_fit(hits);

#if GPU_DEBUG
  printf("KernelFullFitAllHits fast_fit(0): %d %f\n", helix_start, fast_fit(0));
  printf("KernelFullFitAllHits fast_fit(1): %d %f\n", helix_start, fast_fit(1));
  printf("KernelFullFitAllHits fast_fit(2): %d %f\n", helix_start, fast_fit(2));
  printf("KernelFullFitAllHits fast_fit(3): %d %f\n", helix_start, fast_fit(3));
#endif

  u_int n = hits.cols();
#if GPU_DEBUG
  printf("KernelFullFitAllHits using %d hits: %d\n", n, helix_start);
#endif

  Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::circle_fit circle = 
    Rfit::Circle_fit(hits.block(0,0,2,n), hits_cov.block(0, 0, 2 * n, 2 * n),
      fast_fit, rad, B, true, true);

#if GPU_DEBUG
  printf("KernelFullFitAllHits circle.par(0): %d %f\n", helix_start, circle.par(0));
  printf("KernelFullFitAllHits circle.par(1): %d %f\n", helix_start, circle.par(1));
  printf("KernelFullFitAllHits circle.par(2): %d %f\n", helix_start, circle.par(2));
#endif

  Rfit::line_fit line = Rfit::Line_fit(hits, hits_cov, circle, fast_fit, true);

  par_uvrtopak(circle, B, true);

  // Grab helix_fit from the proper location in the output vector
  Rfit::helix_fit &helix = results[helix_start];
  helix.par << circle.par, line.par;
  // TODO: pass properly error booleans
  if (1) {
    helix.cov = MatrixXd::Zero(5, 5);
    helix.cov.block(0, 0, 3, 3) = circle.cov;
    helix.cov.block(3, 3, 2, 2) = line.cov;
  }
  helix.q = circle.q;
  helix.chi2_circle = circle.chi2;
  helix.chi2_line = line.chi2;
}

void PixelTrackReconstructionGPU::launchKernelFit(float * hits_and_covariancesGPU, 
    int cumulative_size, int hits_in_fit, float B, Rfit::helix_fit * results) {
  const dim3 threads_per_block(32,1);
  // We need to partition data in blocks of:
  // 12(3+9) * hits_in_fit
  int num_blocks = cumulative_size/(hits_in_fit*12)/threads_per_block.x + 1;
  KernelFullFitAllHits<<<num_blocks, threads_per_block>>>(hits_and_covariancesGPU, hits_in_fit, cumulative_size, B, results);
  cudaCheck(cudaGetLastError());
}
