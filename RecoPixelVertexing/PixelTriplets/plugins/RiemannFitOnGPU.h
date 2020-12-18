//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = pixelTrack::HitContainer;
using OutputSoA = pixelTrack::TrackSoA;

template <int N>
__global__ void kernelFastFit(Tuples const *__restrict__ foundNtuplets,
                              CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                              uint32_t nHits,
                              HitsOnGPU const *__restrict__ hhp,
                              double *__restrict__ phits,
                              float *__restrict__ phits_ge,
                              double *__restrict__ pfast_fit,
                              uint32_t offset) {
  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit <= nHits);

  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef RIEMANN_DEBUG
  if (0 == local_start)
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
#endif

  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
    assert(tkid < foundNtuplets->nbins());

    assert(foundNtuplets->size(tkid) == nHits);

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    // Prepare data structure
    auto const *hitId = foundNtuplets->begin(tkid);
    for (unsigned int i = 0; i < hitsInFit; ++i) {
      auto hit = hitId[i];
      // printf("Hit global: %f,%f,%f\n", hhp->xg_d[hit],hhp->yg_d[hit],hhp->zg_d[hit]);
      float ge[6];
      hhp->cpeParams()
          .detParams(hhp->detectorIndex(hit))
          .frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
      // printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

      hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
      hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
    }
    Rfit::Fast_fit(hits, fast_fit);

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    assert(fast_fit(3) == fast_fit(3));
  }
}

template <int N>
__global__ void kernelCircleFit(CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                uint32_t nHits,
                                double B,
                                double *__restrict__ phits,
                                float *__restrict__ phits_ge,
                                double *__restrict__ pfast_fit_input,
                                Rfit::circle_fit *circle_fit,
                                uint32_t offset) {
  assert(circle_fit);
  assert(N <= nHits);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit_input + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

    Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
    Rfit::loadCovariance2D(hits_ge, hits_cov);

    circle_fit[local_idx] = Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit, rad, B, true);

#ifdef RIEMANN_DEBUG
//    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
//  printf("kernelCircleFit circle.par(0,1,2): %d %f,%f,%f\n", tkid,
//         circle_fit[local_idx].par(0), circle_fit[local_idx].par(1), circle_fit[local_idx].par(2));
#endif
  }
}

template <int N>
__global__ void kernelLineFit(CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                              uint32_t nHits,
                              double B,
                              OutputSoA *results,
                              double *__restrict__ phits,
                              float *__restrict__ phits_ge,
                              double *__restrict__ pfast_fit_input,
                              Rfit::circle_fit *__restrict__ circle_fit,
                              uint32_t offset) {
  assert(results);
  assert(circle_fit);
  assert(N <= nHits);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it for the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit_input + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    auto const &line_fit = Rfit::Line_fit(hits, hits_ge, circle_fit[local_idx], fast_fit, B, true);

    Rfit::fromCircleToPerigee(circle_fit[local_idx]);

    results->stateAtBS.copyFromCircle(
        circle_fit[local_idx].par, circle_fit[local_idx].cov, line_fit.par, line_fit.cov, 1.f / float(B), tkid);
    results->pt(tkid) = B / std::abs(circle_fit[local_idx].par(2));
    results->eta(tkid) = asinhf(line_fit.par(0));
    results->chi2(tkid) = (circle_fit[local_idx].chi2 + line_fit.chi2) / (2 * N - 5);

#ifdef RIEMANN_DEBUG
    printf("kernelLineFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
           N,
           nHits,
           tkid,
           circle_fit[local_idx].par(0),
           circle_fit[local_idx].par(1),
           circle_fit[local_idx].par(2));
    printf("kernelLineFit line.par(0,1): %d %f,%f\n", tkid, line_fit.par(0), line_fit.par(1));
    printf("kernelLineFit chi2 cov %f/%f %e,%e,%e,%e,%e\n",
           circle_fit[local_idx].chi2,
           line_fit.chi2,
           circle_fit[local_idx].cov(0, 0),
           circle_fit[local_idx].cov(1, 1),
           circle_fit[local_idx].cov(2, 2),
           line_fit.cov(0, 0),
           line_fit.cov(1, 1));
#endif
  }
}
