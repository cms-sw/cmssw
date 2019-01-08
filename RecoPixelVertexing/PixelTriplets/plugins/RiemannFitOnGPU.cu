//
// Author: Felice Pantaleo, CERN
//

#include "RiemannFitOnGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"


using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

using namespace Eigen;

__global__
void kernelFastFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    HitsOnGPU const * __restrict__ hhp,
    int hits_in_fit,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    uint32_t offset)
{

  assert(hits_in_fit==4); // FixMe later template

  assert(pfast_fit); assert(foundNtuplets);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  Rfit::Map3x4d hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6x4f hits_ge(phits_ge+local_start);

  // Prepare data structure
  auto const * hitId = foundNtuplets->begin(helix_start);
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    auto hit = hitId[i];
    // printf("Hit global: %f,%f,%f\n", hhp->xg_d[hit],hhp->yg_d[hit],hhp->zg_d[hit]);
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
    // printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

    hits.col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];
    hits_ge.col(i) << ge[0],ge[1],ge[2],ge[3],ge[4],ge[5];
  }
  Rfit::Fast_fit(hits,fast_fit);

  // no NaN here....
  assert(fast_fit(0)==fast_fit(0));
  assert(fast_fit(1)==fast_fit(1));
  assert(fast_fit(2)==fast_fit(2));
  assert(fast_fit(3)==fast_fit(3));

}

__global__
void kernelCircleFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    int hits_in_fit,
    double B,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit_input,
    Rfit::circle_fit *circle_fit,
    uint32_t offset)
{
  assert(circle_fit); 

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  Rfit::Map3x4d hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit_input+local_start);
  Rfit::Map6x4f hits_ge(phits_ge+local_start);

  constexpr uint32_t N = Rfit::Map3x4d::ColsAtCompileTime;
  constexpr auto n = N;

  assert(4==n); // later will be templated...

  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov =  Rfit::Matrix2Nd<4>::Zero();
  Rfit::loadCovariance2D(hits_ge,hits_cov);

  circle_fit[local_start] =
      Rfit::Circle_fit(hits.block(0, 0, 2, n),
                       hits_cov,
                       fast_fit, rad, B, true);

#ifdef GPU_DEBUG
//  printf("kernelCircleFitAllHits circle.par(0,1,2): %d %f,%f,%f\n", helix_start, 
//         circle_fit[local_start].par(0), circle_fit[local_start].par(1), circle_fit[local_start].par(2));
#endif
}

__global__
void kernelLineFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    int hits_in_fit,
    double B,
    Rfit::helix_fit *results,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    Rfit::circle_fit * __restrict__ circle_fit,
    uint32_t offset)
{

  assert(results); assert(circle_fit);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  Rfit::Map3x4d hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6x4f hits_ge(phits_ge+local_start);
  auto const & line_fit = Rfit::Line_fit(hits, hits_ge, circle_fit[local_start], fast_fit, B, true);

  par_uvrtopak(circle_fit[local_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle_fit[local_start].par, line_fit.par;

  // TODO: pass properly error booleans

  helix.cov = Rfit::Matrix5d::Zero();
  helix.cov.block(0, 0, 3, 3) = circle_fit[local_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit.cov;

  helix.q = circle_fit[local_start].q;
  helix.chi2_circle = circle_fit[local_start].chi2;
  helix.chi2_line = line_fit.chi2;

#ifdef GPU_DEBUG
  printf("kernelLineFitAllHits circle.par(0,1,2): %d %f,%f,%f\n", helix_start,
         circle_fit[local_start].par(0), circle_fit[local_start].par(1), circle_fit[local_start].par(2));
  printf("kernelLineFitAllHits line.par(0,1): %d %f,%f\n", helix_start, line_fit.par(0),line_fit.par(1));
  printf("kernelLineFitAllHits chi2 cov %f/%f %f,%f,%f,%f,%f\n",helix.chi2_circle,helix.chi2_line, 
         helix.cov(0,0),helix.cov(1,1),helix.cov(2,2),helix.cov(3,3),helix.cov(4,4));
#endif
}


void RiemannFitOnGPU::launchKernels(HitsOnCPU const & hh, uint32_t nhits, uint32_t maxNumberOfTuples, cudaStream_t cudaStream)
{
    assert(tuples_d); assert(fast_fit_resultsGPU_);

    auto blockSize = 128;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

    for (uint32_t offset=0; offset<maxNumberOfTuples; offset+=maxNumberOfConcurrentFits_) {
      kernelFastFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, hh.gpu_d, 4,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, 4, bField_,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_, circle_fit_resultsGPU_, offset);
      cudaCheck(cudaGetLastError());


      kernelLineFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tuples_d, 4,  bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_, circle_fit_resultsGPU_,
             offset);
      cudaCheck(cudaGetLastError());
    }
}
