//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = pixelTrack::HitContainer;
using OutputSoA = pixelTrack::TrackSoA;

using namespace Eigen;

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
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);

#ifdef RIEMANN_DEBUG
  if (0 == local_start)
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
#endif

  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits))
    return;

  // get it from the ntuple container (one to one to helix)
  auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_start);
  assert(tkid < foundNtuplets->nbins());

  assert(foundNtuplets->size(tkid) == nHits);

  Rfit::Map3xNd<N> hits(phits + local_start);
  Rfit::Map4d fast_fit(pfast_fit + local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge + local_start);

  // Prepare data structure
  auto const *hitId = foundNtuplets->begin(tkid);
  for (unsigned int i = 0; i < hitsInFit; ++i) {
    auto hit = hitId[i];
    // printf("Hit global: %f,%f,%f\n", hhp->xg_d[hit],hhp->yg_d[hit],hhp->zg_d[hit]);
    float ge[6];
    hhp->cpeParams().detParams(hhp->detectorIndex(hit)).frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
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
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits))
    return;

  // get it for the ntuple container (one to one to helix)
  auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_start);

  Rfit::Map3xNd<N> hits(phits + local_start);
  Rfit::Map4d fast_fit(pfast_fit_input + local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge + local_start);

  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge, hits_cov);

  circle_fit[local_start] = Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit, rad, B, true);

#ifdef RIEMANN_DEBUG
//  printf("kernelCircleFit circle.par(0,1,2): %d %f,%f,%f\n", tkid,
//         circle_fit[local_start].par(0), circle_fit[local_start].par(1), circle_fit[local_start].par(2));
#endif
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
  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits))
    return;

  // get it for the ntuple container (one to one to helix)
  auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_start);

  Rfit::Map3xNd<N> hits(phits + local_start);
  Rfit::Map4d fast_fit(pfast_fit_input + local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge + local_start);

  auto const &line_fit = Rfit::Line_fit(hits, hits_ge, circle_fit[local_start], fast_fit, B, true);

  Rfit::fromCircleToPerigee(circle_fit[local_start]);

  results->stateAtBS.copyFromCircle(circle_fit[local_start].par,circle_fit[local_start].cov,
                                   line_fit.par,line_fit.cov,1.f/float(B),tkid);
  results->pt(tkid) =  B/std::abs(circle_fit[local_start].par(2));
  results->eta(tkid) =  asinhf(line_fit.par(0));
  results->chi2(tkid) = (circle_fit[local_start].chi2+line_fit.chi2)/(2*N-5);

#ifdef RIEMANN_DEBUG
  printf("kernelLineFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
         N,
         nHits,
         tkid,
         circle_fit[local_start].par(0),
         circle_fit[local_start].par(1),
         circle_fit[local_start].par(2));
  printf("kernelLineFit line.par(0,1): %d %f,%f\n", tkid, line_fit.par(0), line_fit.par(1));
  printf("kernelLineFit chi2 cov %f/%f %e,%e,%e,%e,%e\n",
         circle_fit[local_start].chi2,
         line_fit.chi2,
         circle_fit[local_start].cov(0, 0),
         circle_fit[local_start].cov(1, 1),
         circle_fit[local_start].cov(2, 2),
         line_fit.cov(0, 0),
         line_fit.cov(1, 1));
#endif
}

void HelixFitOnGPU::launchRiemannKernels(HitsOnCPU const &hh,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples,
                                         cuda::stream_t<> &stream) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  edm::Service<CUDAService> cs;
  auto hitsGPU_ = cs->make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ =
      cs->make_device_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ =
      cs->make_device_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);
  auto circle_fit_resultsGPU_holder =
      cs->make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                    tupleMultiplicity_d,
                                                                    3,
                                                                    hh.view(),
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<3><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                      3,
                                                                      bField_,
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<3><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                    3,
                                                                    bField_,
                                                                    outputSoa_d,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    circle_fit_resultsGPU_,
                                                                    offset);
    cudaCheck(cudaGetLastError());

    // quads
    kernelFastFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                    tupleMultiplicity_d,
                                                                    4,
                                                                    hh.view(),
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                      4,
                                                                      bField_,
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                    4,
                                                                    bField_,
                                                                    outputSoa_d,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    circle_fit_resultsGPU_,
                                                                    offset);
    cudaCheck(cudaGetLastError());

    if (fit5as4_) {
      // penta
      kernelFastFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                      tupleMultiplicity_d,
                                                                      5,
                                                                      hh.view(),
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                        5,
                                                                        bField_,
                                                                        hitsGPU_.get(),
                                                                        hits_geGPU_.get(),
                                                                        fast_fit_resultsGPU_.get(),
                                                                        circle_fit_resultsGPU_,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                      5,
                                                                      bField_,
                                                                      outputSoa_d,
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
      cudaCheck(cudaGetLastError());
    } else {
      // penta all 5
      kernelFastFit<5><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                      tupleMultiplicity_d,
                                                                      5,
                                                                      hh.view(),
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<5><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                        5,
                                                                        bField_,
                                                                        hitsGPU_.get(),
                                                                        hits_geGPU_.get(),
                                                                        fast_fit_resultsGPU_.get(),
                                                                        circle_fit_resultsGPU_,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<5><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                      5,
                                                                      bField_,
                                                                      outputSoa_d,
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
      cudaCheck(cudaGetLastError());
    }
  }
}
