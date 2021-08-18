//
// Author: Felice Pantaleo, CERN
//

// #define BROKENLINE_DEBUG

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"

#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = pixelTrack::HitContainer;
using OutputSoA = pixelTrack::TrackSoA;

// #define BL_DUMP_HITS

template <int N>
__global__ void kernel_BLFastFit(Tuples const *__restrict__ foundNtuplets,
                                 caConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                 HitsOnGPU const *__restrict__ hhp,
                                 double *__restrict__ phits,
                                 float *__restrict__ phits_ge,
                                 double *__restrict__ pfast_fit,
                                 uint32_t nHits,
                                 uint32_t offset) {
  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit <= nHits);

  assert(hhp);
  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef BROKENLINE_DEBUG
  if (0 == local_start) {
    printf("%d total Ntuple\n", foundNtuplets->nOnes());
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
  }
#endif

  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
    assert(tkid < foundNtuplets->nOnes());

    assert(foundNtuplets->size(tkid) == nHits);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#ifdef BL_DUMP_HITS
    __shared__ int done;
    done = 0;
    __syncthreads();
    bool dump = (foundNtuplets->size(tkid) == 5 && 0 == atomicAdd(&done, 1));
#endif

    // Prepare data structure
    auto const *hitId = foundNtuplets->begin(tkid);

    // #define YERR_FROM_DC
#ifdef YERR_FROM_DC
    // try to compute more precise error in y
    auto dx = hhp->xGlobal(hitId[hitsInFit - 1]) - hhp->xGlobal(hitId[0]);
    auto dy = hhp->yGlobal(hitId[hitsInFit - 1]) - hhp->yGlobal(hitId[0]);
    auto dz = hhp->zGlobal(hitId[hitsInFit - 1]) - hhp->zGlobal(hitId[0]);
    float ux, uy, uz;
#endif
    for (unsigned int i = 0; i < hitsInFit; ++i) {
      auto hit = hitId[i];
      float ge[6];
#ifdef YERR_FROM_DC
      auto const &dp = hhp->cpeParams().detParams(hhp->detectorIndex(hit));
      auto status = hhp->status(hit);
      int qbin = 4 - status.qBin;
      assert(qbin >= 0 && qbin < 5);
      bool nok = (status.isBigY | status.isOneY);
      // compute cotanbeta and use it to recompute error
      dp.frame.rotation().multiply(dx, dy, dz, ux, uy, uz);
      auto cb = std::abs(uy / uz);
      int bin = int(cb * (285.f / 150.f) * 8.f) - 4;
      bin = std::max(0, std::min(15, bin));
      float yerr = dp.sigmay[bin] * 1.e-4f;
      yerr *= dp.yfact[qbin];  // inflate
      yerr *= yerr;
      yerr += dp.apeYY;
      yerr = nok ? hhp->yerrLocal(hit) : yerr;
      dp.frame.toGlobal(hhp->xerrLocal(hit), 0, yerr, ge);
#else
      hhp->cpeParams()
          .detParams(hhp->detectorIndex(hit))
          .frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
#endif

#ifdef BL_DUMP_HITS
      if (dump) {
        printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n",
               tkid,
               hhp->detectorIndex(hit),
               i,
               hhp->xGlobal(hit),
               hhp->yGlobal(hit),
               hhp->zGlobal(hit));
        printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",
               tkid,
               hhp->detetectorIndex(hit),
               i,
               ge[0],
               ge[1],
               ge[2],
               ge[3],
               ge[4],
               ge[5]);
      }
#endif
      hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
      hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
    }
    brokenline::fastFit(hits, fast_fit);

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    assert(fast_fit(3) == fast_fit(3));
  }
}

template <int N>
__global__ void kernel_BLFit(caConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                             double bField,
                             OutputSoA *results,
                             double *__restrict__ phits,
                             float *__restrict__ phits_ge,
                             double *__restrict__ pfast_fit,
                             uint32_t nHits,
                             uint32_t offset) {
  assert(N <= nHits);

  assert(results);
  assert(pfast_fit);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it for the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    brokenline::PreparedBrokenLineData<N> data;

    brokenline::karimaki_circle_fit circle;
    riemannFit::LineFit line;

    brokenline::prepareBrokenLineData(hits, fast_fit, bField, data);
    brokenline::lineFit(hits_ge, fast_fit, bField, data, line);
    brokenline::circleFit(hits, hits_ge, fast_fit, bField, data, circle);

    results->stateAtBS.copyFromCircle(circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
    results->pt(tkid) = float(bField) / float(std::abs(circle.par(2)));
    results->eta(tkid) = asinhf(line.par(0));
    results->chi2(tkid) = (circle.chi2 + line.chi2) / (2 * N - 5);

#ifdef BROKENLINE_DEBUG
    if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
      printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
    printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
           N,
           nHits,
           tkid,
           circle.par(0),
           circle.par(1),
           circle.par(2));
    printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
    printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
           circle.chi2,
           line.chi2,
           circle.cov(0, 0),
           circle.cov(1, 1),
           circle.cov(2, 2),
           line.cov(0, 0),
           line.cov(1, 1));
#endif
  }
}
