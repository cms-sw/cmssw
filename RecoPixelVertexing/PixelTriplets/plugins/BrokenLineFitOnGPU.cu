//
// Author: Felice Pantaleo, CERN
//

#include "HelixFitOnGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"

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


// #define BL_DUMP_HITS

template<int N>
__global__
void kernelBLFastFit(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    CAConstants::TupleMultiplicity const * __restrict__ tupleMultiplicity,
    HitsOnGPU const * __restrict__ hhp,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    uint32_t nHits,
    uint32_t offset)
{

  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit<=nHits);

  assert(pfast_fit); assert(foundNtuplets);

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);

#ifdef BROKENLINE_DEBUG
  if (0==local_start) printf("%d Ntuple of size %d for %d hits to fit\n",tupleMultiplicity->size(nHits), nHits, hitsInFit);
#endif

  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits)) return;

  // get it from the ntuple container (one to one to helix)
  auto helix_start = *(tupleMultiplicity->begin(nHits)+tuple_start);
  assert (helix_start < foundNtuplets->nbins());

  assert (foundNtuplets->size(helix_start)==nHits);

  Rfit::Map3xNd<N> hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge+local_start);

#ifdef BL_DUMP_HITS
  __shared__ int done;
  done = 0;
  __syncthreads(); 
  bool dump =  (foundNtuplets->size(helix_start)==5 &&
                0 == atomicAdd(&done,1));
#endif

  // Prepare data structure
  auto const * hitId = foundNtuplets->begin(helix_start);
  for (unsigned int i = 0; i < hitsInFit; ++i) {
    auto hit = hitId[i];
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
#ifdef BL_DUMP_HITS
    if (dump){
      printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n", helix_start, hhp->detInd_d[hit],i,hhp->xg_d[hit],hhp->yg_d[hit],hhp->zg_d[hit]);
      printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",helix_start,hhp->detInd_d[hit],i,ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);
    }
#endif
    hits.col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];
    hits_ge.col(i) << ge[0],ge[1],ge[2],ge[3],ge[4],ge[5];
  }
  BrokenLine::BL_Fast_fit(hits,fast_fit);

  // no NaN here....
  assert(fast_fit(0)==fast_fit(0));
  assert(fast_fit(1)==fast_fit(1));
  assert(fast_fit(2)==fast_fit(2));
  assert(fast_fit(3)==fast_fit(3));

} 

template<int N>
__global__
void kernelBLFit(
    CAConstants::TupleMultiplicity const * __restrict__ tupleMultiplicity,
    double B,
    Rfit::helix_fit *results,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    uint32_t nHits,
    uint32_t offset)
{

  assert(N<=nHits);

  assert(results); assert(pfast_fit);


  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits)) return;

  // get it for the ntuple container (one to one to helix)
  auto helix_start = *(tupleMultiplicity->begin(nHits)+tuple_start);


  Rfit::Map3xNd<N> hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge+local_start);

  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;

  BrokenLine::karimaki_circle_fit circle;
  Rfit::line_fit line;
 
  BrokenLine::prepareBrokenLineData(hits,fast_fit,B,data);
  BrokenLine::BL_Line_fit(hits_ge,fast_fit,B,data,line);
  BrokenLine::BL_Circle_fit(hits,hits_ge,fast_fit,B,data,circle);
  Jacob << 1,0,0,
    0,1,0,
    0,0,-B/std::copysign(Rfit::sqr(circle.par(2)),circle.par(2));
  circle.par(2)=B/std::abs(circle.par(2));
  circle.cov=Jacob*circle.cov*Jacob.transpose();


  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle.par, line.par;

  helix.cov = Rfit::Matrix5d::Zero();
  helix.cov.block(0, 0, 3, 3) = circle.cov;
  helix.cov.block(3, 3, 2, 2) = line.cov;

  helix.q = circle.q;
  helix.chi2_circle = circle.chi2;
  helix.chi2_line = line.chi2;

#ifdef BROKENLINE_DEBUG
  if ( !(circle.chi2>=0) || !(line.chi2>=0) ) printf("kernelBLFit failed! %f/%f\n", helix.chi2_circle,helix.chi2_line);
  printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n", N,nHits, helix_start,
         circle.par(0), circle.par(1), circle.par(2));
  printf("kernelBLHits line.par(0,1): %d %f,%f\n", helix_start, line.par(0),line.par(1));
  printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",helix.chi2_circle,helix.chi2_line, 
         helix.cov(0,0),helix.cov(1,1),helix.cov(2,2),helix.cov(3,3),helix.cov(4,4));
#endif
}


void HelixFitOnGPU::launchBrokenLineKernels(HitsOnCPU const & hh, uint32_t hitsInFit, uint32_t maxNumberOfTuples, cudaStream_t cudaStream)
{
    assert(tuples_d); assert(fast_fit_resultsGPU_);

    auto blockSize = 64;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

    for (uint32_t offset=0; offset<maxNumberOfTuples; offset+=maxNumberOfConcurrentFits_) {

      // fit triplets
      kernelBLFastFit<3><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, tupleMultiplicity_d, hh.gpu_d,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
          3, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<3><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tupleMultiplicity_d, bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
             3, offset);
      cudaCheck(cudaGetLastError());

      // fit quads
      kernelBLFastFit<4><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, tupleMultiplicity_d, hh.gpu_d,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
          4, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<4><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tupleMultiplicity_d, bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
             4, offset);
      cudaCheck(cudaGetLastError());

      if (fit5as4_) {
        // fit penta (only first 4)
        kernelBLFastFit<4><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, tupleMultiplicity_d, hh.gpu_d,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
          5, offset);
        cudaCheck(cudaGetLastError());

        kernelBLFit<4><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tupleMultiplicity_d, bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
             5, offset);
        cudaCheck(cudaGetLastError());
      } else {
        // fit penta (all 5)
        kernelBLFastFit<5><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, tupleMultiplicity_d, hh.gpu_d,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
          5, offset);
        cudaCheck(cudaGetLastError());

        kernelBLFit<5><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tupleMultiplicity_d, bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
             5, offset);
        cudaCheck(cudaGetLastError());
      }

    } // loop on concurrent fits


}
