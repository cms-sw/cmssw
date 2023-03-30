//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "HelixFitOnGPU.h"

template <typename TrackerTraits>
using Tuples = typename TrackSoA<TrackerTraits>::HitContainer;
template <typename TrackerTraits>
using OutputSoAView = TrackSoAView<TrackerTraits>;
template <typename TrackerTraits>
using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

template <int N, typename TrackerTraits>
__global__ void kernel_FastFit(Tuples<TrackerTraits> const *__restrict__ foundNtuplets,
                               TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                               uint32_t nHits,
                               TrackingRecHitSoAConstView<TrackerTraits> hh,
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

  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
    assert(int(tkid) < foundNtuplets->nOnes());

    assert(foundNtuplets->size(tkid) == nHits);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    // Prepare data structure
    auto const *hitId = foundNtuplets->begin(tkid);
    for (unsigned int i = 0; i < hitsInFit; ++i) {
      auto hit = hitId[i];
      float ge[6];
      hh.cpeParams().detParams(hh[hit].detectorIndex()).frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);

      hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
      hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
    }
    riemannFit::fastFit(hits, fast_fit);

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    assert(fast_fit(3) == fast_fit(3));
  }
}

template <int N, typename TrackerTraits>
__global__ void kernel_CircleFit(TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                                 uint32_t nHits,
                                 double bField,
                                 double *__restrict__ phits,
                                 float *__restrict__ phits_ge,
                                 double *__restrict__ pfast_fit_input,
                                 riemannFit::CircleFit *circle_fit,
                                 uint32_t offset) {
  assert(circle_fit);
  assert(N <= nHits);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    riemannFit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

    riemannFit::Matrix2Nd<N> hits_cov = riemannFit::Matrix2Nd<N>::Zero();
    riemannFit::loadCovariance2D(hits_ge, hits_cov);

    circle_fit[local_idx] = riemannFit::circleFit(hits.block(0, 0, 2, N), hits_cov, fast_fit, rad, bField, true);

#ifdef RIEMANN_DEBUG
//    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
//  printf("kernelCircleFit circle.par(0,1,2): %d %f,%f,%f\n", tkid,
//         circle_fit[local_idx].par(0), circle_fit[local_idx].par(1), circle_fit[local_idx].par(2));
#endif
  }
}

template <int N, typename TrackerTraits>
__global__ void kernel_LineFit(TupleMultiplicity<TrackerTraits> const *__restrict__ tupleMultiplicity,
                               uint32_t nHits,
                               double bField,
                               OutputSoAView<TrackerTraits> results_view,
                               double *__restrict__ phits,
                               float *__restrict__ phits_ge,
                               double *__restrict__ pfast_fit_input,
                               riemannFit::CircleFit *__restrict__ circle_fit,
                               uint32_t offset) {
  assert(circle_fit);
  assert(N <= nHits);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  for (int local_idx = local_start, nt = riemannFit::maxNumberOfConcurrentFits; local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it for the ntuple container (one to one to helix)
    int32_t tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

    riemannFit::Map3xNd<N> hits(phits + local_idx);
    riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
    riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    auto const &line_fit = riemannFit::lineFit(hits, hits_ge, circle_fit[local_idx], fast_fit, bField, true);

    riemannFit::fromCircleToPerigee(circle_fit[local_idx]);

    TracksUtilities<TrackerTraits>::copyFromCircle(results_view,
                                                   circle_fit[local_idx].par,
                                                   circle_fit[local_idx].cov,
                                                   line_fit.par,
                                                   line_fit.cov,
                                                   1.f / float(bField),
                                                   tkid);
    results_view[tkid].pt() = bField / std::abs(circle_fit[local_idx].par(2));
    results_view[tkid].eta() = asinhf(line_fit.par(0));
    results_view[tkid].chi2() = (circle_fit[local_idx].chi2 + line_fit.chi2) / (2 * N - 5);

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
