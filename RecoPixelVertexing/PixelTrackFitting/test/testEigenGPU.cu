#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "test_common.h"

using namespace Eigen;

__global__
void kernelFullFit(Rfit::Matrix3xNd * hits,
    Rfit::Matrix3Nd * hits_cov,
    double B,
    bool errors,
    Rfit::circle_fit * circle_fit_resultsGPU,
    Rfit::line_fit * line_fit_resultsGPU) {

  printf("hits size: %d,%d\n", hits->rows(), hits->cols());
  Rfit::printIt(hits, "KernelFulFit - input hits: ");
  Vector4d fast_fit = Rfit::Fast_fit(*hits);

  u_int n = hits->cols();
  Rfit::VectorNd rad = (hits->block(0, 0, 2, n).colwise().norm());

  Rfit::Matrix2xNd hits2D_local = (hits->block(0,0,2,n)).eval();
  Rfit::Matrix2Nd hits_cov2D_local = (hits_cov->block(0, 0, 2 * n, 2 * n)).eval();
  Rfit::printIt(&hits2D_local, "kernelFullFit - hits2D_local: ");
  Rfit::printIt(&hits_cov2D_local, "kernelFullFit - hits_cov2D_local: ");
  printf("kernelFullFit - hits address: %p\n", hits);
  printf("kernelFullFit - hits_cov address: %p\n", hits_cov);
  printf("kernelFullFit - hits_cov2D address: %p\n", &hits2D_local);
  printf("kernelFullFit - hits_cov2D_local address: %p\n", &hits_cov2D_local);
  /* At some point I gave up and locally construct block on the stack, so that
     the next invocation to Rfit::Circle_fit works properly. Failing to do so
     implied basically an empty collection of hits and covariances. That could
     have been partially fixed if values of the passed in matrices would have
     been printed on screen since that, maybe, triggered internally the real
     creations of the blocks. To be understood and compared against the myriad
     of compilation warnings we have.
     */

  (*circle_fit_resultsGPU) =
    Rfit::Circle_fit(hits->block(0,0,2,n), hits_cov->block(0, 0, 2 * n, 2 * n),
        fast_fit, rad, B, errors);
  /*
     (*circle_fit_resultsGPU) =
     Rfit::Circle_fit(hits2D_local, hits_cov2D_local,
     fast_fit, rad, B, errors);
  */
  (*line_fit_resultsGPU) = Rfit::Line_fit(*hits, *hits_cov, *circle_fit_resultsGPU, fast_fit, B, errors);

  return;
}

void fillHitsAndHitsCov(Rfit::Matrix3xNd & hits, Rfit::Matrix3Nd & hits_cov) {
  hits << 1.98645, 4.72598, 7.65632, 11.3151,
          2.18002, 4.88864, 7.75845, 11.3134,
          2.46338, 6.99838,  11.808,  17.793;
  hits_cov(0,0) = 7.14652e-06;
  hits_cov(1,1) = 2.15789e-06;
  hits_cov(2,2) = 1.63328e-06;
  hits_cov(3,3) = 6.27919e-06;
  hits_cov(4,4) = 6.10348e-06;
  hits_cov(5,5) = 2.08211e-06;
  hits_cov(6,6) = 1.61672e-06;
  hits_cov(7,7) = 6.28081e-06;
  hits_cov(8,8) = 5.184e-05;
  hits_cov(9,9) = 1.444e-05;
  hits_cov(10,10) = 6.25e-06;
  hits_cov(11,11) = 3.136e-05;
  hits_cov(0,4) = hits_cov(4,0) = -5.60077e-06;
  hits_cov(1,5) = hits_cov(5,1) = -1.11936e-06;
  hits_cov(2,6) = hits_cov(6,2) = -6.24945e-07;
  hits_cov(3,7) = hits_cov(7,3) = -5.28e-06;
}

void testFitOneGo(bool errors, double epsilon=1e-6) {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd hits(3,4);
  Rfit::Matrix3Nd hits_cov = MatrixXd::Zero(12,12);

  fillHitsAndHitsCov(hits, hits_cov);

  // FAST_FIT_CPU
  Vector4d fast_fit_results = Rfit::Fast_fit(hits);
  // CIRCLE_FIT CPU
  u_int n = hits.cols();
  Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
      hits_cov.block(0, 0, 2 * n, 2 * n),
      fast_fit_results, rad, B, errors);
  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_cov, circle_fit_results,
      fast_fit_results, B, errors);

  // FIT GPU
  std::cout << "GPU FIT" << std::endl;
  Rfit::Matrix3xNd * hitsGPU = nullptr; // new Rfit::Matrix3xNd(3,4);
  Rfit::Matrix3Nd * hits_covGPU = nullptr;
  Rfit::line_fit * line_fit_resultsGPU = nullptr;
  Rfit::line_fit * line_fit_resultsGPUret = new Rfit::line_fit();
  Rfit::circle_fit * circle_fit_resultsGPU = nullptr; // new Rfit::circle_fit();
  Rfit::circle_fit * circle_fit_resultsGPUret = new Rfit::circle_fit();

  cudaCheck(cudaMalloc((void **)&hitsGPU, sizeof(Rfit::Matrix3xNd(3,4))));
  cudaCheck(cudaMalloc((void **)&hits_covGPU, sizeof(Rfit::Matrix3Nd(12,12))));
  cudaCheck(cudaMalloc((void **)&line_fit_resultsGPU, sizeof(Rfit::line_fit)));
  cudaCheck(cudaMalloc((void **)&circle_fit_resultsGPU, sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemcpy(hitsGPU, &hits, sizeof(Rfit::Matrix3xNd(3,4)), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(hits_covGPU, &hits_cov, sizeof(Rfit::Matrix3Nd(12,12)), cudaMemcpyHostToDevice));

  kernelFullFit<<<1, 1>>>(hitsGPU, hits_covGPU, B, errors,
      circle_fit_resultsGPU, line_fit_resultsGPU);
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU, sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost));

  std::cout << "Fitted values (CircleFit) CPU:\n" << circle_fit_results.par << std::endl;
  std::cout << "Fitted values (LineFit): CPU\n" << line_fit_results.par << std::endl;
  std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << std::endl;
  std::cout << "Fitted values (LineFit): GPU\n" << line_fit_resultsGPUret->par << std::endl;
  assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par, epsilon));
  assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par, epsilon));

  cudaCheck(cudaFree(hitsGPU));
  cudaCheck(cudaFree(hits_covGPU));
  cudaCheck(cudaFree(line_fit_resultsGPU));
  cudaCheck(cudaFree(circle_fit_resultsGPU));
  delete line_fit_resultsGPUret;
  delete circle_fit_resultsGPUret;

  cudaDeviceReset();
}

int main (int argc, char * argv[]) {

  cudaDeviceSetLimit(cudaLimitStackSize, 32*1024);
  std::cout << "TEST FIT, ERRORS AND SCATTER" << std::endl;
  testFitOneGo(true, 1e-5);

  return 0;
}

