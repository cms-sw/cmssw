//
// Author: Felice Pantaleo, CERN
//

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitQuadrupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

using namespace gpuPixelDoublets;

using HitsOnGPU = CAHitQuadrupletGeneratorKernels::HitsOnGPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
using Quality = pixelTuplesHeterogeneousProduct::Quality;

__global__ void kernel_checkOverflows(TuplesOnGPU::Container *foundNtuplets,
                                      AtomicPairCounter *apc,
                                      GPUCACell const *__restrict__ cells,
                                      uint32_t const *__restrict__ nCells,
                                      CellNeighborsVector const *cellNeighbors,
                                      CellTracksVector const *cellTracks,
                                      GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                                      uint32_t nHits,
                                      CAHitQuadrupletGeneratorKernels::Counters *counters) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;

  auto &c = *counters;
  // counters once per event
  if (0 == idx) {
    atomicAdd(&c.nEvents, 1);
    atomicAdd(&c.nHits, nHits);
    atomicAdd(&c.nCells, *nCells);
    atomicAdd(&c.nTuples, apc->get().m);
  }

#ifdef GPU_DEBUG
  if (0 == idx) {
    printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",
           *nCells,
           apc->get().m,
           apc->get().n,
           nHits);
    assert(foundNtuplets->size(apc->get().m) == 0);
    assert(foundNtuplets->size() == apc->get().n);
  }

  if (idx < foundNtuplets->nbins()) {
    if (foundNtuplets->size(idx) > 5)
      printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
    assert(foundNtuplets->size(idx) < 6);
    for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
      assert(*ih < nHits);
  }
#endif

  if (0 == idx) {
    if (apc->get().m >= CAConstants::maxNumberOfQuadruplets())
      printf("Tuples overflow\n");
    if (*nCells >= CAConstants::maxNumberOfDoublets())
      printf("Cells overflow\n");
  }

  if (idx < (*nCells)) {
    auto &thisCell = cells[idx];
    if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
      printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
      printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.theDoubletId < 0)
      atomicAdd(&c.nKilledCells, 1);
    if (thisCell.outerNeighbors().empty())
      atomicAdd(&c.nEmptyCells, 1);
    if (thisCell.tracks().empty())
      atomicAdd(&c.nZeroTrackCells, 1);
  }
  if (idx < nHits) {
    if (isOuterHitOfCell[idx].full())  // ++tooManyOuterHitOfCell;
      printf("OuterHitOfCell overflow %d\n", idx);
  }
}

__global__ void kernel_fishboneCleaner(GPUCACell const *cells,
                                       uint32_t const *__restrict__ nCells,
                                       pixelTuplesHeterogeneousProduct::Quality *quality) {
  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId >= 0)
    return;

  for (auto it : thisCell.tracks())
    quality[it] = bad;
}

__global__ void kernel_fastDuplicateRemover(GPUCACell const *cells,
                                            uint32_t const *__restrict__ nCells,
                                            TuplesOnGPU::Container *foundNtuplets,
                                            Rfit::helix_fit const *__restrict__ hfit,
                                            pixelTuplesHeterogeneousProduct::Quality *quality) {
  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;
  constexpr auto dup = pixelTuplesHeterogeneousProduct::dup;
  // constexpr auto loose = pixelTuplesHeterogeneousProduct::loose;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId < 0)
    return;

  float mc = 1000.f;
  uint16_t im = 60000;
  uint32_t maxNh = 0;

  auto score = [&](auto it) {
    return std::abs(hfit[it].par(1));  // tip
    // return hfit[it].chi2_line+hfit[it].chi2_circle;  //chi2
  };

  // find maxNh
  for (auto it : thisCell.tracks()) {
    if (quality[it] == bad)
      continue;
    auto nh = foundNtuplets->size(it);
    maxNh = std::max(nh, maxNh);
  }
  // find min chi2
  for (auto it : thisCell.tracks()) {
    auto nh = foundNtuplets->size(it);
    if (nh != maxNh)
      continue;
    if (quality[it] != bad && score(it) < mc) {
      mc = score(it);
      im = it;
    }
  }
  // mark duplicates
  for (auto it : thisCell.tracks()) {
    if (quality[it] != bad && it != im)
      quality[it] = dup;  //no race:  simple assignment of the same constant
  }
}

__global__ void kernel_connect(AtomicPairCounter *apc1,
                               AtomicPairCounter *apc2,  // just to zero them,
                               GPUCACell::Hits const *__restrict__ hhp,
                               GPUCACell *cells,
                               uint32_t const *__restrict__ nCells,
                               CellNeighborsVector *cellNeighbors,
                               GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                               float hardCurvCut,
                               float ptmin,
                               float CAThetaCutBarrel,
                               float CAThetaCutForward,
                               float dcaCutInnerTriplet,
                               float dcaCutOuterTriplet) {
  auto const &hh = *hhp;

  auto cellIndex = threadIdx.y + blockIdx.y * blockDim.y;
  auto first = threadIdx.x;
  auto stride = blockDim.x;

  if (0 == (cellIndex + first)) {
    (*apc1) = 0;
    (*apc2) = 0;
  }  // ready for next kernel

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId < 0)
    return;
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  auto vi = isOuterHitOfCell[innerHitId].data();
  for (auto j = first; j < numberOfPossibleNeighbors; j += stride) {
    auto otherCell = __ldg(vi + j);
    if (cells[otherCell].theDoubletId < 0)
      continue;
    if (thisCell.check_alignment(hh,
                                 cells[otherCell],
                                 ptmin,
                                 hardCurvCut,
                                 CAThetaCutBarrel,
                                 CAThetaCutForward,
                                 dcaCutInnerTriplet,
                                 dcaCutOuterTriplet)) {
      cells[otherCell].addOuterNeighbor(cellIndex, *cellNeighbors);
    }
  }
}

__global__ void kernel_find_ntuplets(GPUCACell::Hits const *__restrict__ hhp,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells,
                                     CellTracksVector *cellTracks,
                                     TuplesOnGPU::Container *foundNtuplets,
                                     AtomicPairCounter *apc,
                                     GPUCACell::TupleMultiplicity *tupleMultiplicity,
                                     unsigned int minHitsPerNtuplet) {
  // recursive: not obvious to widen
  auto const &hh = *hhp;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells))
    return;
  auto &thisCell = cells[cellIndex];

#ifdef CA_USE_LOCAL_COUNTERS
  __shared__ GPUCACell::TupleMultiplicity::CountersOnly local;
  if (0 == threadIdx.x)
    local.zero();
  __syncthreads();
#endif

  if (thisCell.theLayerPairId == 0 || thisCell.theLayerPairId == 3 ||
      thisCell.theLayerPairId == 8) {  // inner layer is 0 FIXME
    GPUCACell::TmpTuple stack;
    stack.reset();
    thisCell.find_ntuplets(hh,
                           cells,
                           *cellTracks,
                           *foundNtuplets,
                           *apc,
#ifdef CA_USE_LOCAL_COUNTERS
                           local,
#else
                           *tupleMultiplicity,
#endif
                           stack,
                           minHitsPerNtuplet);
    assert(stack.size() == 0);
    // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
  }

#ifdef CA_USE_LOCAL_COUNTERS
  __syncthreads();
  if (0 == threadIdx.x)
    tupleMultiplicity->add(local);
#endif
}

__global__ void kernel_fillMultiplicity(TuplesOnGPU::Container const *__restrict__ foundNtuplets,
                                        GPUCACell::TupleMultiplicity *tupleMultiplicity) {
  auto it = blockIdx.x * blockDim.x + threadIdx.x;

  if (it >= foundNtuplets->nbins())
    return;

  auto nhits = foundNtuplets->size(it);
  if (nhits < 3)
    return;
  tupleMultiplicity->fillDirect(nhits, it);
}

__global__ void kernel_classifyTracks(TuplesOnGPU::Container const *__restrict__ tuples,
                                      Rfit::helix_fit const *__restrict__ fit_results,
                                      CAHitQuadrupletGeneratorKernels::QualityCuts cuts,
                                      Quality *__restrict__ quality) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tuples->nbins()) {
    return;
  }
  if (tuples->size(idx) == 0) {
    return;
  }

  // mark as bad by default
  quality[idx] = pixelTuplesHeterogeneousProduct::bad;

  // mark doublets as bad
  if (tuples->size(idx) < 3) {
    return;
  }

  // if the fit has any invalid parameters, mark it as bad
  bool isNaN = false;
  for (int i = 0; i < 5; ++i) {
    isNaN |= isnan(fit_results[idx].par(i));
  }
  if (isNaN) {
#ifdef GPU_DEBUG
    printf("NaN in fit %d size %d chi2 %f + %f\n",
           idx,
           tuples->size(idx),
           fit_results[idx].chi2_line,
           fit_results[idx].chi2_circle);
#endif
    return;
  }

  // compute a pT-dependent chi2 cut
  // default parameters:
  //   - chi2MaxPt = 10 GeV
  //   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
  //   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
  // (see CAHitQuadrupletGeneratorGPU.cc)
  float pt = std::min<float>(fit_results[idx].par(2), cuts.chi2MaxPt);
  float chi2Cut = cuts.chi2Scale *
                  (cuts.chi2Coeff[0] + pt * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3])));
  if (fit_results[idx].chi2_line + fit_results[idx].chi2_circle >= chi2Cut) {
#ifdef GPU_DEBUG
    printf("Bad fit %d size %d chi2 %f + %f\n",
           idx,
           tuples->size(idx),
           fit_results[idx].chi2_line,
           fit_results[idx].chi2_circle);
#endif
    return;
  }

  // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
  // default cuts:
  //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
  //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
  // (see CAHitQuadrupletGeneratorGPU.cc)
  auto const &region = (tuples->size(idx) > 3) ? cuts.quadruplet : cuts.triplet;
  bool isOk = (std::abs(fit_results[idx].par(1)) < region.maxTip) and (fit_results[idx].par(2) > region.minPt) and
              (std::abs(fit_results[idx].par(4)) < region.maxZip);

  if (isOk) {
    quality[idx] = pixelTuplesHeterogeneousProduct::loose;
  }
}

__global__ void kernel_doStatsForTracks(TuplesOnGPU::Container const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitQuadrupletGeneratorKernels::Counters *counters) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != pixelTuplesHeterogeneousProduct::loose)
      continue;
    atomicAdd(&(counters->nGoodTracks), 1);
  }
}

__global__ void kernel_countHitInTracks(TuplesOnGPU::Container const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitQuadrupletGeneratorKernels::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != pixelTuplesHeterogeneousProduct::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  }
}

__global__ void kernel_fillHitInTracks(TuplesOnGPU::Container const *__restrict__ tuples,
                                       Quality const *__restrict__ quality,
                                       CAHitQuadrupletGeneratorKernels::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != pixelTuplesHeterogeneousProduct::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h, idx);
  }
}

__global__ void kernel_doStatsForHitInTracks(CAHitQuadrupletGeneratorKernels::HitToTuple const *__restrict__ hitToTuple,
                                             CAHitQuadrupletGeneratorKernels::Counters *counters) {
  auto &c = *counters;
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple->size(idx) == 0)
      continue;
    atomicAdd(&c.nUsedHits, 1);
    if (hitToTuple->size(idx) > 1)
      atomicAdd(&c.nDupHits, 1);
  }
}

__global__ void kernel_tripletCleaner(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                      TuplesOnGPU::Container const *__restrict__ ptuples,
                                      Rfit::helix_fit const *__restrict__ hfit,
                                      Quality *__restrict__ quality,
                                      CAHitQuadrupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple) {
  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;
  constexpr auto dup = pixelTuplesHeterogeneousProduct::dup;
  // constexpr auto loose = pixelTuplesHeterogeneousProduct::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &foundNtuplets = *ptuples;

  //  auto const & hh = *hhp;
  // auto l1end = hh.hitsLayerStart_d[1];

  int first = blockDim.x * blockIdx.x + threadIdx.x;

  for (int idx = first, ntot = hitToTuple.nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = 10000.f;
    uint16_t im = 60000;
    uint32_t maxNh = 0;

    // find maxNh
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      maxNh = std::max(nh, maxNh);
    }
    // kill all tracks shorter than maxHn (only triplets???)
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      if (maxNh != nh)
        quality[*it] = dup;
    }

    if (maxNh > 3)
      continue;
    // if (idx>=l1end) continue;  // only for layer 1
    // for triplets choose best tip!
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && std::abs(hfit[it].par(1)) < mc) {
        mc = std::abs(hfit[it].par(1));
        im = it;
      }
    }
    // mark duplicates
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && it != im)
        quality[it] = dup;  //no race:  simple assignment of the same constant
    }
  }  // loop over hits
}

__global__ void kernel_print_found_ntuplets(TuplesOnGPU::Container const *__restrict__ foundNtuplets,
                                            uint32_t maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->nbins()); ++i) {
    if (foundNtuplets->size(i) < 4)
      continue;
    printf("\nquadruplet %d: %d %d %d %d\n",
           i,
           (*(*foundNtuplets).begin(i)),
           (*(*foundNtuplets).begin(i) + 1),
           (*(*foundNtuplets).begin(i) + 2),
           (*(*foundNtuplets).begin(i) + 3));
  }
}

void CAHitQuadrupletGeneratorKernels::launchKernels(  // here goes algoparms....
    HitsOnCPU const &hh,
    TuplesOnGPU &tuples_d,
    cudaStream_t cudaStream) {
  auto &gpu_ = tuples_d;
  auto maxNumberOfDoublets_ = CAConstants::maxNumberOfDoublets();

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  if (nhits > 1 && earlyFishbone_) {
    auto nthTot = 64;
    auto stride = 4;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
    cudaCheck(cudaGetLastError());
  }

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot / stride;
  auto numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1) / blockSize;
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1) / blockSize;
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  kernel_connect<<<blks, thrs, 0, cudaStream>>>(
      gpu_.apc_d,
      device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
      hh.view(),
      device_theCells_.get(),
      device_nCells_,
      device_theCellNeighbors_,
      device_isOuterHitOfCell_.get(),
      hardCurvCut_,
      ptmin_,
      CAThetaCutBarrel_,
      CAThetaCutForward_,
      dcaCutInnerTriplet_,
      dcaCutOuterTriplet_);
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_,
                                                                     gpu_.tuples_d,
                                                                     gpu_.apc_d,
                                                                     device_tupleMultiplicity_,
                                                                     minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  numberOfBlocks = (TuplesOnGPU::Container::totbins() + blockSize - 1) / blockSize;
  cudautils::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.apc_d, gpu_.tuples_d);

  cudautils::launchFinalize(device_tupleMultiplicity_, device_tmws_, cudaStream);

  blockSize = 128;
  numberOfBlocks = (CAConstants::maxTuples() + blockSize - 1) / blockSize;
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.tuples_d, device_tupleMultiplicity_);
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
    cudaCheck(cudaGetLastError());
  }

  if (doStats_) {
    numberOfBlocks = (std::max(nhits, maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.tuples_d,
                                                                        gpu_.apc_d,
                                                                        device_theCells_.get(),
                                                                        device_nCells_,
                                                                        device_theCellNeighbors_,
                                                                        device_theCellTracks_,
                                                                        device_isOuterHitOfCell_.get(),
                                                                        nhits,
                                                                        counters_);
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
#endif
  }

  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(gpu_.tuples_d, 10);
}

void CAHitQuadrupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, cuda::stream_t<> &stream) {
  auto nhits = hh.nHits();

#ifdef GPU_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  edm::Service<CUDAService> cs;
  device_isOuterHitOfCell_ = cs->make_device_unique<GPUCACell::OuterHitOfCell[]>(nhits, stream);
  {
    int threadsPerBlock = 128;
    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0, stream.id()>>>(device_isOuterHitOfCell_.get(),
                                                                                nhits,
                                                                                device_theCellNeighbors_,
                                                                                device_theCellNeighborsContainer_.get(),
                                                                                device_theCellTracks_,
                                                                                device_theCellTracksContainer_.get());
    cudaCheck(cudaGetLastError());
  }

  device_theCells_ = cs->make_device_unique<GPUCACell[]>(CAConstants::maxNumberOfDoublets(), stream);

  if (0 == nhits)
    return;  // protect against empty events

  int stride = 1;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (2 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream.id()>>>(device_theCells_.get(),
                                                                         device_nCells_,
                                                                         device_theCellNeighbors_,
                                                                         device_theCellTracks_,
                                                                         hh.view(),
                                                                         device_isOuterHitOfCell_.get(),
                                                                         idealConditions_,
                                                                         doClusterCut_,
                                                                         doZCut_,
                                                                         doPhiCut_);
  cudaCheck(cudaGetLastError());
}

void CAHitQuadrupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh,
                                                     TuplesOnGPU &tuples,
                                                     cudaStream_t cudaStream) {
  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples.tuples_d, tuples.helix_fit_results_d, cuts_, tuples.quality_d);
  cudaCheck(cudaGetLastError());

  // apply fishbone cleaning to good tracks
  numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1) / blockSize;
  kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples.quality_d);
  cudaCheck(cudaGetLastError());

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1) / blockSize;
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples.tuples_d, tuples.helix_fit_results_d, tuples.quality_d);
  cudaCheck(cudaGetLastError());

  // fill hit->track "map"
  numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
  kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples.tuples_d, tuples.quality_d, device_hitToTuple_);
  cudaCheck(cudaGetLastError());
  cudautils::launchFinalize(device_hitToTuple_, device_tmws_, cudaStream);
  cudaCheck(cudaGetLastError());
  kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples.tuples_d, tuples.quality_d, device_hitToTuple_);
  cudaCheck(cudaGetLastError());

  // remove duplicates (tracks that share a hit)
  numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
  kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      hh.view(), tuples.tuples_d, tuples.helix_fit_results_d, tuples.quality_d, device_hitToTuple_);
  cudaCheck(cudaGetLastError());

  if (doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_, counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d, tuples.quality_d, counters_);
    cudaCheck(cudaGetLastError());
  }
}

__global__ void kernel_printCounters(CAHitQuadrupletGeneratorKernels::Counters const *counters) {
  auto const &c = *counters;
  printf(
      "||Counters | nEvents | nHits | nCells | nTuples | nGoodTracks | nUsedHits | nDupHits | nKilledCells | "
      "nEmptyCells | nZeroTrackCells ||\n");
  printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
         c.nEvents,
         c.nHits,
         c.nCells,
         c.nTuples,
         c.nGoodTracks,
         c.nUsedHits,
         c.nDupHits,
         c.nKilledCells,
         c.nEmptyCells,
         c.nZeroTrackCells);
  printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f||\n",
         c.nEvents,
         c.nHits / double(c.nEvents),
         c.nCells / double(c.nEvents),
         c.nTuples / double(c.nEvents),
         c.nGoodTracks / double(c.nEvents),
         c.nUsedHits / double(c.nEvents),
         c.nDupHits / double(c.nEvents),
         c.nKilledCells / double(c.nEvents),
         c.nEmptyCells / double(c.nCells),
         c.nZeroTrackCells / double(c.nCells));
}

void CAHitQuadrupletGeneratorKernels::printCounters() const { kernel_printCounters<<<1, 1>>>(counters_); }
