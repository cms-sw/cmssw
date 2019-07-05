//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

using namespace gpuPixelDoublets;

  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

__global__ void kernel_checkOverflows(HitContainer const * foundNtuplets,
                                      CAConstants::TupleMultiplicity * tupleMultiplicity,
                                      AtomicPairCounter *apc,
                                      GPUCACell const *__restrict__ cells,
                                      uint32_t const *__restrict__ nCells,
                                      CellNeighborsVector const *cellNeighbors,
                                      CellTracksVector const *cellTracks,
                                      GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                                      uint32_t nHits,
                                      CAHitNtupletGeneratorKernels::Counters *counters) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;

  auto &c = *counters;
  // counters once per event
  if (0 == idx) {
    atomicAdd(&c.nEvents, 1);
    atomicAdd(&c.nHits, nHits);
    atomicAdd(&c.nCells, *nCells);
    atomicAdd(&c.nTuples, apc->get().m);
    atomicAdd(&c.nFitTracks,tupleMultiplicity->size());
  }

#ifdef NTUPLE_DEBUG
  if (0 == idx) {
    printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",
           *nCells,
           apc->get().m,
           apc->get().n,
           nHits);
    if (apc->get().m < CAConstants::maxNumberOfQuadruplets()) {
      assert(foundNtuplets->size(apc->get().m) == 0);
      assert(foundNtuplets->size() == apc->get().n);
    }
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
    if (0==thisCell.theUsed)
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
                                       Quality *quality) {
  constexpr auto bad = trackQuality::bad;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId >= 0)
    return;

  for (auto it : thisCell.tracks())
    quality[it] = bad;
}

__global__ void kernel_earlyDuplicateRemover(GPUCACell const *cells,
                                            uint32_t const *__restrict__ nCells,
                                            HitContainer *foundNtuplets,
                                            Quality *quality) {
  // constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;

  assert(nCells);

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId < 0)
    return;

  uint32_t maxNh = 0;

  // find maxNh
  for (auto it : thisCell.tracks()) {
    auto nh = foundNtuplets->size(it);
    maxNh = std::max(nh, maxNh);
  }

  for (auto it : thisCell.tracks()) {
    if (foundNtuplets->size(it) != maxNh)
      quality[it] = dup;  //no race:  simple assignment of the same constant
  }

}


__global__ void kernel_fastDuplicateRemover(GPUCACell const * __restrict__ cells,
                                            uint32_t const *__restrict__ nCells,
                                            HitContainer const * __restrict__ foundNtuplets,
                                            TkSoA * __restrict__ tracks) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  constexpr auto loose = trackQuality::loose;

  assert(nCells);

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells))
    return;
  auto const &thisCell = cells[cellIndex];
  if (thisCell.theDoubletId < 0)
    return;

  float mc = 10000.f;
  uint16_t im = 60000;

  auto score = [&](auto it) {
    return std::abs(tracks->tip(it));  // tip
    // return tracks->chi2(it);  //chi2
  };

  // find min socre
  for (auto it : thisCell.tracks()) {
    if (tracks->quality(it) == loose && score(it) < mc) {
      mc = score(it);
      im = it;
    }
  }
  // mark all other duplicates
  for (auto it : thisCell.tracks()) {
    if (tracks->quality(it) != bad && it != im)
        tracks->quality(it) = dup;  //no race:  simple assignment of the same constant
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
  auto & thisCell = cells[cellIndex];
  //if (thisCell.theDoubletId < 0 || thisCell.theUsed>1)
  //  return;
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  auto vi = isOuterHitOfCell[innerHitId].data();

  constexpr uint32_t last_bpix1_detIndex = 96;
  constexpr uint32_t last_barrel_detIndex = 1184;
  auto ri = thisCell.get_inner_r(hh);
  auto zi = thisCell.get_inner_z(hh);

  auto ro = thisCell.get_outer_r(hh);
  auto zo = thisCell.get_outer_z(hh);
  auto isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex;

  for (auto j = first; j < numberOfPossibleNeighbors; j += stride) {
    auto otherCell = __ldg(vi + j);
    auto & oc = cells[otherCell];
    // if (cells[otherCell].theDoubletId < 0 ||
    //    cells[otherCell].theUsed>1 )
    //  continue;
    auto r1 = oc.get_inner_r(hh);
    auto z1 = oc.get_inner_z(hh);
    // auto isBarrel = oc.get_outer_detIndex(hh) < last_barrel_detIndex;
    bool aligned = GPUCACell::areAlignedRZ(r1,
                                z1,
                                ri,
                                zi,
                                ro,
                                zo,
                                ptmin,
                                isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
    if(aligned &&
      thisCell.dcaCut(hh,oc,
                      oc.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet : dcaCutOuterTriplet,
                      hardCurvCut)
       ) {  // FIXME tune cuts
      oc.addOuterNeighbor(cellIndex, *cellNeighbors);
      thisCell.theUsed |= 1;
      oc.theUsed |= 1;
    }
  } // loop on inner cells
}

__global__ void kernel_find_ntuplets(GPUCACell::Hits const *__restrict__ hhp,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells,
                                     CellTracksVector *cellTracks,
                                     HitContainer *foundNtuplets,
                                     AtomicPairCounter *apc,
                                     Quality * __restrict__ quality,
                                     unsigned int minHitsPerNtuplet) {
  // recursive: not obvious to widen
  auto const &hh = *hhp;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells))
    return;
  auto &thisCell = cells[cellIndex];

  if (thisCell.theDoubletId < 0)
    return;

  auto pid = thisCell.theLayerPairId;
  auto doit = minHitsPerNtuplet>3 ? pid<3 : pid<8 || pid >12;
  if (doit) {
    GPUCACell::TmpTuple stack;
    stack.reset();
    thisCell.find_ntuplets(hh,
                           cells,
                           *cellTracks,
                           *foundNtuplets,
                           *apc,
                           quality,
                           stack,
                           minHitsPerNtuplet, 
                           pid<3);
    assert(stack.size() == 0);
    // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
  }

}


__global__ void kernel_mark_used(GPUCACell::Hits const *__restrict__ hhp,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells) {

  // auto const &hh = *hhp;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells))
    return;
  auto &thisCell = cells[cellIndex];
  if (!thisCell.tracks().empty())
    thisCell.theUsed |= 2;

}


__global__ void kernel_countMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                         Quality const * __restrict__ quality,
                                         CAConstants::TupleMultiplicity *tupleMultiplicity) {
  auto it = blockIdx.x * blockDim.x + threadIdx.x;

  if (it >= foundNtuplets->nbins())
    return;

  auto nhits = foundNtuplets->size(it);
  if (nhits < 3)
    return;
  if (quality[it] == trackQuality::dup) return;
  assert(quality[it] == trackQuality::bad);
  if (nhits>5) printf("wrong mult %d %d\n",it,nhits);
  assert(nhits<8);
  tupleMultiplicity->countDirect(nhits);
}



__global__ void kernel_fillMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                        Quality const * __restrict__ quality,
                                        CAConstants::TupleMultiplicity *tupleMultiplicity) {
  auto it = blockIdx.x * blockDim.x + threadIdx.x;

  if (it >= foundNtuplets->nbins())
    return;

  auto nhits = foundNtuplets->size(it);
  if (nhits < 3)
    return;
  if (quality[it] == trackQuality::dup) return;
  if (nhits>5) printf("wrong mult %d %d\n",it,nhits);
  assert(nhits<8);
  tupleMultiplicity->fillDirect(nhits, it);
}



__global__ void kernel_classifyTracks(HitContainer const *__restrict__ tuples,
                                      TkSoA const * __restrict__ tracks,
                                      CAHitNtupletGeneratorKernels::QualityCuts cuts,
                                      Quality *__restrict__ quality) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tuples->nbins()) {
    return;
  }
  if (tuples->size(idx) == 0) {
    return;
  }

  // if duplicate: not even fit
  if (quality[idx] == trackQuality::dup) return;

  assert(quality[idx] == trackQuality::bad);

  // mark doublets as bad
  if (tuples->size(idx) < 3) {
    return;
  }

  // if the fit has any invalid parameters, mark it as bad
  bool isNaN = false;
  for (int i = 0; i < 5; ++i) {
    isNaN |= isnan(tracks->stateAtBS.state(idx)(i));
  }
  if (isNaN) {
#ifdef NTUPLE_DEBUG
    printf("NaN in fit %d size %d chi2 %f\n",
           idx,
           tuples->size(idx),
           tracks->chi2(idx)
    );
#endif
    return;
  }

  // compute a pT-dependent chi2 cut
  // default parameters:
  //   - chi2MaxPt = 10 GeV
  //   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
  //   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
  // (see CAHitNtupletGeneratorGPU.cc)
  float pt = std::min<float>(tracks->pt(idx), cuts.chi2MaxPt);
  float chi2Cut = cuts.chi2Scale *
                  (cuts.chi2Coeff[0] + pt * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3])));
  // above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
  if (3.f*tracks->chi2(idx) >= chi2Cut) {
#ifdef NTUPLE_DEBUG
    printf("Bad fit %d size %d pt %f eta %f chi2 %f\n",
           idx,
           tuples->size(idx), 
           tracks->pt(idx),
           tracks->eta(idx),
           3.f*tracks->chi2(idx)
    );
#endif
    return;
  }

  // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
  // default cuts:
  //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
  //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
  // (see CAHitNtupletGeneratorGPU.cc)
  auto const &region = (tuples->size(idx) > 3) ? cuts.quadruplet : cuts.triplet;
  bool isOk = (std::abs(tracks->tip(idx)) < region.maxTip) and (tracks->pt(idx) > region.minPt) and
              (std::abs(tracks->zip(idx)) < region.maxZip);

  if (isOk) {
    quality[idx] = trackQuality::loose;
  }
}

__global__ void kernel_doStatsForTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernels::Counters *counters) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != trackQuality::loose)
      continue;
    atomicAdd(&(counters->nGoodTracks), 1);
  }
}


__global__ void kernel_countHitInTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != trackQuality::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  }
}

__global__ void kernel_fillHitInTracks(HitContainer const *__restrict__ tuples,
                                       Quality const *__restrict__ quality,
                                       CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      continue;
    if (quality[idx] != trackQuality::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h, idx);
  }
}

__global__ void kernel_fillHitDetIndices(HitContainer const *__restrict__ tuples,
                                         TrackingRecHit2DSOAView const *__restrict__ hhp,
                                         HitContainer *__restrict__ hitDetIndices) {

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  // copy offsets
  for (int idx = first, ntot = tuples->totbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
    hitDetIndices->off[idx] = tuples->off[idx];
  }
  // fill hit indices
  auto const & hh = *hhp;
  auto nhits = hh.nHits();
  for (int idx = first, ntot = tuples->size(); idx < ntot; idx += gridDim.x * blockDim.x) {
    assert(tuples->bins[idx]<nhits);
    hitDetIndices->bins[idx] = hh.detectorIndex(tuples->bins[idx]);
  }
}

void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsOnCPU const &hh, TkSoA * tracks_d, cudaStream_t cudaStream) {
  auto blockSize=128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;

  kernel_fillHitDetIndices<<<numberOfBlocks,blockSize,0,cudaStream>>>(&tracks_d->hitIndices, hh.view(), &tracks_d->detIndices);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif
}


__global__ void kernel_doStatsForHitInTracks(CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ hitToTuple,
                                             CAHitNtupletGeneratorKernels::Counters *counters) {
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
                                      HitContainer const *__restrict__ ptuples,
                                      TkSoA const * __restrict__ ptracks,
                                      Quality *__restrict__ quality,
                                      CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &foundNtuplets = *ptuples;
  auto const & tracks = *ptracks;

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
      if (quality[it] != bad && std::abs(tracks.tip(it)) < mc) {
        mc = std::abs(tracks.tip(it));
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

__global__ void kernel_print_found_ntuplets(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                      HitContainer const *__restrict__ ptuples,
                                      TkSoA const * __restrict__ ptracks,
                                      Quality const *__restrict__ quality,
                                      CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple,
                                      uint32_t maxPrint, int iev) {
  auto const & foundNtuplets = *ptuples;
  auto const & tracks = *ptracks;
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < std::min(maxPrint, foundNtuplets.nbins()); i+=blockDim.x*gridDim.x) {
    auto nh = foundNtuplets.size(i);
    if (nh<3) continue;
    printf("TK: %d %d %d %f %f %f %f %f %f %f %d %d %d %d %d\n",
           10000*iev+i,
           int(quality[i]),
           nh,
           tracks.charge(i),
           tracks.pt(i),
           tracks.eta(i),
           tracks.phi(i),
           tracks.tip(i),
           tracks.zip(i),
//           asinhf(fit_results[i].par(3)),
           tracks.chi2(i),
           *foundNtuplets.begin(i),
           *(foundNtuplets.begin(i) + 1),
           *(foundNtuplets.begin(i) + 2),
           nh>3 ? int(*(foundNtuplets.begin(i) + 3)):-1,
           nh>4 ? int(*(foundNtuplets.begin(i) + 4)):-1
          );
  }
}


void CAHitNtupletGeneratorKernels::launchKernels(
    HitsOnCPU const &hh,
    TkSoA * tracks_d,
    cudaStream_t cudaStream) {

  auto maxNumberOfDoublets_ = CAConstants::maxNumberOfDoublets();

  // these are pointer on GPU!
  auto * tuples_d = &tracks_d->hitIndices; 
  auto * quality_d = (Quality*)(&tracks_d->m_quality);  

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

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
      device_hitTuple_apc_,
      device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
      hh.view(),
      device_theCells_.get(),
      device_nCells_,
      device_theCellNeighbors_,
      device_isOuterHitOfCell_.get(),
      m_params.hardCurvCut_,
      m_params.ptmin_,
      m_params.CAThetaCutBarrel_,
      m_params.CAThetaCutForward_,
      m_params.dcaCutInnerTriplet_,
      m_params.dcaCutOuterTriplet_);
  cudaCheck(cudaGetLastError());


  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
    cudaCheck(cudaGetLastError());
  }


  blockSize = 64;
  numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1) / blockSize;
  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_,
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     m_params.minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  if (m_params.doStats_)
    kernel_mark_used<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                   device_theCells_.get(),
                                                                   device_nCells_);
  cudaCheck(cudaGetLastError());
  
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif


  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  cudautils::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1) / blockSize;
  kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, quality_d);
  cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (CAConstants::maxTuples() + blockSize - 1) / blockSize;
  kernel_countMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudautils::launchFinalize(device_tupleMultiplicity_.get(), device_tmws_, cudaStream);
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.lateFishbone_) {
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

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d,
                                                                        device_tupleMultiplicity_.get(),
                                                                        device_hitTuple_apc_,
                                                                        device_theCells_.get(),
                                                                        device_nCells_,
                                                                        device_theCellNeighbors_,
                                                                        device_theCellTracks_,
                                                                        device_isOuterHitOfCell_.get(),
                                                                        nhits,
                                                                        counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

}

void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, cuda::stream_t<> &stream) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  edm::Service<CUDAService> cs;
  device_isOuterHitOfCell_ = cs->make_device_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U,nhits), stream);
  assert(device_isOuterHitOfCell_.get());
  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = ( std::max(1U,nhits) + threadsPerBlock - 1) / threadsPerBlock;
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0, stream.id()>>>(device_isOuterHitOfCell_.get(),
                                                                                nhits,
                                                                                device_theCellNeighbors_,
                                                                                device_theCellNeighborsContainer_.get(),
                                                                                device_theCellTracks_,
                                                                                device_theCellTracksContainer_.get());
    cudaCheck(cudaGetLastError());
  }

  device_theCells_ = cs->make_device_unique<GPUCACell[]>(CAConstants::maxNumberOfDoublets(), stream);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  if (0 == nhits)
    return;  // protect against empty events

  // FIXME avoid magic numbers
  auto nActualPairs=gpuPixelDoublets::nPairs;
  if (!m_params.includeJumpingForwardDoublets_) nActualPairs = 15;
  if (m_params.minHitsPerNtuplet_>3) {
    nActualPairs = 13;
  }

  assert(nActualPairs<=gpuPixelDoublets::nPairs);
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
                                                                         nActualPairs,
                                                                         m_params.idealConditions_,
                                                                         m_params.doClusterCut_,
                                                                         m_params.doZCut_,
                                                                         m_params.doPhiCut_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

}

void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh,
                                                     TkSoA * tracks_d,
                                                     cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto const * tuples_d = &tracks_d->hitIndices;
  auto * quality_d = (Quality*)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples_d, tracks_d, m_params.cuts_, quality_d);
  cudaCheck(cudaGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1) / blockSize;
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        device_theCells_.get(), device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1) / blockSize;
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, tracks_d);
  cudaCheck(cudaGetLastError());

  if (m_params.minHitsPerNtuplet_<4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
    kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    cudautils::launchFinalize(device_hitToTuple_.get(), device_tmws_, cudaStream);
    cudaCheck(cudaGetLastError());
    kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
  }
  if (m_params.minHitsPerNtuplet_<4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
  }
  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_.get(), counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

#ifdef    DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets<<<1, 32, 0, cudaStream>>>(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100,iev);
#endif

}

__global__ void kernel_printCounters(CAHitNtupletGeneratorKernels::Counters const *counters) {
  auto const &c = *counters;
  printf(
      "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nGoodTracks | nUsedHits | nDupHits | nKilledCells | "
      "nEmptyCells | nZeroTrackCells ||\n");
  printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
         c.nEvents,
         c.nHits,
         c.nCells,
         c.nTuples,
         c.nGoodTracks,
         c.nFitTracks,
         c.nUsedHits,
         c.nDupHits,
         c.nKilledCells,
         c.nEmptyCells,
         c.nZeroTrackCells);
  printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f||\n",
         c.nEvents,
         c.nHits / double(c.nEvents),
         c.nCells / double(c.nEvents),
         c.nTuples / double(c.nEvents),
         c.nFitTracks / double(c.nEvents),
         c.nGoodTracks / double(c.nEvents),
         c.nUsedHits / double(c.nEvents),
         c.nDupHits / double(c.nEvents),
         c.nKilledCells / double(c.nEvents),
         c.nEmptyCells / double(c.nCells),
         c.nZeroTrackCells / double(c.nCells));
}

void CAHitNtupletGeneratorKernels::printCounters(Counters const * counters) { 
   kernel_printCounters<<<1, 1>>>(counters);
}


