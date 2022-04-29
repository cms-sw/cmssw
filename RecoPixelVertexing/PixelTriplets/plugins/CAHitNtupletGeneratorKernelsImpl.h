//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG
// #define GPU_DEBUG

#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using HitsOnCPU = TrackingRecHit2DGPU;

using HitToTuple = caConstants::HitToTuple;
using TupleMultiplicity = caConstants::TupleMultiplicity;

using Quality = pixelTrack::Quality;
using TkSoA = pixelTrack::TrackSoA;
using HitContainer = pixelTrack::HitContainer;

namespace {

  constexpr uint16_t tkNotFound = std::numeric_limits<uint16_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

}  // namespace

__global__ void kernel_checkOverflows(HitContainer const *foundNtuplets,
                                      caConstants::TupleMultiplicity const *tupleMultiplicity,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *hitToTuple,
                                      cms::cuda::AtomicPairCounter *apc,
                                      GPUCACell const *__restrict__ cells,
                                      uint32_t const *__restrict__ nCells,
                                      gpuPixelDoublets::CellNeighborsVector const *cellNeighbors,
                                      gpuPixelDoublets::CellTracksVector const *cellTracks,
                                      GPUCACell::OuterHitOfCell const isOuterHitOfCell,
                                      int32_t nHits,
                                      uint32_t maxNumberOfDoublets,
                                      CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto first = threadIdx.x + blockIdx.x * blockDim.x;

  auto &c = *counters;
  // counters once per event
  if (0 == first) {
    atomicAdd(&c.nEvents, 1);
    atomicAdd(&c.nHits, nHits);
    atomicAdd(&c.nCells, *nCells);
    atomicAdd(&c.nTuples, apc->get().m);
    atomicAdd(&c.nFitTracks, tupleMultiplicity->size());
  }

#ifdef NTUPLE_DEBUG
  if (0 == first) {
    printf("number of found cells %d, found tuples %d with total hits %d out of %d %d\n",
           *nCells,
           apc->get().m,
           apc->get().n,
           nHits,
           hitToTuple->totOnes());
    if (apc->get().m < caConstants::maxNumberOfQuadruplets) {
      assert(foundNtuplets->size(apc->get().m) == 0);
      assert(foundNtuplets->size() == apc->get().n);
    }
  }

  for (int idx = first, nt = foundNtuplets->nOnes(); idx < nt; idx += gridDim.x * blockDim.x) {
    if (foundNtuplets->size(idx) > 7)  // current real limit
      printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
    assert(foundNtuplets->size(idx) <= caConstants::maxHitsOnTrack);
    for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
      assert(int(*ih) < nHits);
  }
#endif

  if (0 == first) {
    if (apc->get().m >= caConstants::maxNumberOfQuadruplets)
      printf("Tuples overflow\n");
    if (*nCells >= maxNumberOfDoublets)
      printf("Cells overflow\n");
    if (cellNeighbors && cellNeighbors->full())
      printf("cellNeighbors overflow\n");
    if (cellTracks && cellTracks->full())
      printf("cellTracks overflow\n");
    if (int(hitToTuple->nOnes()) < nHits)
      printf("ERROR hitToTuple  overflow %d %d\n", hitToTuple->nOnes(), nHits);
  }

  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.hasFishbone() && !thisCell.isKilled())
      atomicAdd(&c.nFishCells, 1);
    if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
      printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.layerPairId());
    if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
      printf("Tracks overflow %d in %d\n", idx, thisCell.layerPairId());
    if (thisCell.isKilled())
      atomicAdd(&c.nKilledCells, 1);
    if (!thisCell.unused())
      atomicAdd(&c.nEmptyCells, 1);
    if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
      atomicAdd(&c.nZeroTrackCells, 1);
  }

  for (int idx = first, nt = nHits - isOuterHitOfCell.offset; idx < nt; idx += gridDim.x * blockDim.x) {
    if (isOuterHitOfCell.container[idx].full())  // ++tooManyOuterHitOfCell;
      printf("OuterHitOfCell overflow %d\n", idx);
  }
}

__global__ void kernel_fishboneCleaner(GPUCACell const *cells, uint32_t const *__restrict__ nCells, Quality *quality) {
  constexpr auto reject = pixelTrack::Quality::dup;

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (!thisCell.isKilled())
      continue;

    for (auto it : thisCell.tracks())
      quality[it] = reject;
  }
}

// remove shorter tracks if sharing a cell
// It does not seem to affect efficiency in any way!
__global__ void kernel_earlyDuplicateRemover(GPUCACell const *cells,
                                             uint32_t const *__restrict__ nCells,
                                             TkSoA const *__restrict__ ptracks,
                                             Quality *quality,
                                             bool dupPassThrough) {
  // quality to mark rejected
  constexpr auto reject = pixelTrack::Quality::edup;  /// cannot be loose

  auto const &tracks = *ptracks;

  assert(nCells);
  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];

    if (thisCell.tracks().size() < 2)
      continue;

    int8_t maxNl = 0;

    // find maxNl
    for (auto it : thisCell.tracks()) {
      auto nl = tracks.nLayers(it);
      maxNl = std::max(nl, maxNl);
    }

    // if (maxNl<4) continue;
    // quad pass through (leave it her for tests)
    //  maxNl = std::min(4, maxNl);

    for (auto it : thisCell.tracks()) {
      if (tracks.nLayers(it) < maxNl)
        quality[it] = reject;  //no race:  simple assignment of the same constant
    }
  }
}

// assume the above (so, short tracks already removed)
__global__ void kernel_fastDuplicateRemover(GPUCACell const *__restrict__ cells,
                                            uint32_t const *__restrict__ nCells,
                                            TkSoA *__restrict__ tracks,
                                            bool dupPassThrough) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
  constexpr auto loose = pixelTrack::Quality::loose;

  assert(nCells);

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.tracks().size() < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;

    /* chi2 penalize higher-pt tracks  (try rescale it?)
    auto score = [&](auto it) {
      return tracks->nLayers(it) < 4 ? 
              std::abs(tracks->tip(it)) :  // tip for triplets
              tracks->chi2(it);            //chi2 for quads
    };
    */

    auto score = [&](auto it) { return std::abs(tracks->tip(it)); };

    // full crazy combinatorics
    int ntr = thisCell.tracks().size();
    for (int i = 0; i < ntr - 1; ++i) {
      auto it = thisCell.tracks()[i];
      auto qi = tracks->quality(it);
      if (qi <= reject)
        continue;
      auto opi = tracks->stateAtBS.state(it)(2);
      auto e2opi = tracks->stateAtBS.covariance(it)(9);
      auto cti = tracks->stateAtBS.state(it)(3);
      auto e2cti = tracks->stateAtBS.covariance(it)(12);
      for (auto j = i + 1; j < ntr; ++j) {
        auto jt = thisCell.tracks()[j];
        auto qj = tracks->quality(jt);
        if (qj <= reject)
          continue;
#ifdef GPU_DEBUG
        if (foundNtuplets->size(it) != foundNtuplets->size(jt))
          printf(" a mess\n");
#endif
        auto opj = tracks->stateAtBS.state(jt)(2);
        auto ctj = tracks->stateAtBS.state(jt)(3);
        auto dct = nSigma2 * (tracks->stateAtBS.covariance(jt)(12) + e2cti);
        if ((cti - ctj) * (cti - ctj) > dct)
          continue;
        auto dop = nSigma2 * (tracks->stateAtBS.covariance(jt)(9) + e2opi);
        if ((opi - opj) * (opi - opj) > dop)
          continue;
        if ((qj < qi) || (qj == qi && score(it) < score(jt)))
          tracks->quality(jt) = reject;
        else {
          tracks->quality(it) = reject;
          break;
        }
      }
    }

    // find maxQual
    auto maxQual = reject;  // no duplicate!
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) > maxQual)
        maxQual = tracks->quality(it);
    }

    if (maxQual <= loose)
      continue;

    // find min score
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) == maxQual && score(it) < mc) {
        mc = score(it);
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark all other duplicates  (not yet, keep it loose)
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) > loose && it != im)
        tracks->quality(it) = loose;  //no race:  simple assignment of the same constant
    }
  }
}

__global__ void kernel_connect(cms::cuda::AtomicPairCounter *apc1,
                               cms::cuda::AtomicPairCounter *apc2,  // just to zero them,
                               GPUCACell::Hits const *__restrict__ hhp,
                               GPUCACell *cells,
                               uint32_t const *__restrict__ nCells,
                               gpuPixelDoublets::CellNeighborsVector *cellNeighbors,
                               GPUCACell::OuterHitOfCell const isOuterHitOfCell,
                               float hardCurvCut,
                               float ptmin,
                               float CAThetaCutBarrel,
                               float CAThetaCutForward,
                               float dcaCutInnerTriplet,
                               float dcaCutOuterTriplet) {
  auto const &hh = *hhp;

  auto firstCellIndex = threadIdx.y + blockIdx.y * blockDim.y;
  auto first = threadIdx.x;
  auto stride = blockDim.x;

  if (0 == (firstCellIndex + first)) {
    (*apc1) = 0;
    (*apc2) = 0;
  }  // ready for next kernel

  for (int idx = firstCellIndex, nt = (*nCells); idx < nt; idx += gridDim.y * blockDim.y) {
    auto cellIndex = idx;
    auto &thisCell = cells[idx];
    auto innerHitId = thisCell.inner_hit_id();
    if (int(innerHitId) < isOuterHitOfCell.offset)
      continue;
    int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
    auto vi = isOuterHitOfCell[innerHitId].data();

    auto ri = thisCell.inner_r(hh);
    auto zi = thisCell.inner_z(hh);

    auto ro = thisCell.outer_r(hh);
    auto zo = thisCell.outer_z(hh);
    auto isBarrel = thisCell.inner_detIndex(hh) < caConstants::last_barrel_detIndex;

    for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
      auto otherCell = __ldg(vi + j);
      auto &oc = cells[otherCell];
      auto r1 = oc.inner_r(hh);
      auto z1 = oc.inner_z(hh);
      bool aligned = GPUCACell::areAlignedRZ(
          r1,
          z1,
          ri,
          zi,
          ro,
          zo,
          ptmin,
          isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
      if (aligned && thisCell.dcaCut(hh,
                                     oc,
                                     oc.inner_detIndex(hh) < caConstants::last_bpix1_detIndex ? dcaCutInnerTriplet
                                                                                              : dcaCutOuterTriplet,
                                     hardCurvCut)) {  // FIXME tune cuts
        oc.addOuterNeighbor(cellIndex, *cellNeighbors);
        thisCell.setStatusBits(GPUCACell::StatusBit::kUsed);
        oc.setStatusBits(GPUCACell::StatusBit::kUsed);
      }
    }  // loop on inner cells
  }    // loop on outer cells
}

__global__ void kernel_find_ntuplets(GPUCACell::Hits const *__restrict__ hhp,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells,
                                     gpuPixelDoublets::CellTracksVector *cellTracks,
                                     HitContainer *foundNtuplets,
                                     cms::cuda::AtomicPairCounter *apc,
                                     Quality *__restrict__ quality,
                                     unsigned int minHitsPerNtuplet) {
  // recursive: not obvious to widen
  auto const &hh = *hhp;

  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto const &thisCell = cells[idx];
    if (thisCell.isKilled())
      continue;  // cut by earlyFishbone
    // we require at least three hits...
    if (thisCell.outerNeighbors().empty())
      continue;
    auto pid = thisCell.layerPairId();
    auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
    if (doit) {
      GPUCACell::TmpTuple stack;
      stack.reset();
      thisCell.find_ntuplets<6>(
          hh, cells, *cellTracks, *foundNtuplets, *apc, quality, stack, minHitsPerNtuplet, pid < 3);
      assert(stack.empty());
      // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
    }
  }
}

__global__ void kernel_mark_used(GPUCACell *__restrict__ cells, uint32_t const *nCells) {
  auto first = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
    auto &thisCell = cells[idx];
    if (!thisCell.tracks().empty())
      thisCell.setStatusBits(GPUCACell::StatusBit::kInTrack);
  }
}

__global__ void kernel_countMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                         Quality const *__restrict__ quality,
                                         caConstants::TupleMultiplicity *tupleMultiplicity) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int it = first, nt = foundNtuplets->nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      continue;
    if (quality[it] == pixelTrack::Quality::edup)
      continue;
    assert(quality[it] == pixelTrack::Quality::bad);
    if (nhits > 7)  // current limit
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits <= caConstants::maxHitsOnTrack);
    tupleMultiplicity->count(nhits);
  }
}

__global__ void kernel_fillMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                        Quality const *__restrict__ quality,
                                        caConstants::TupleMultiplicity *tupleMultiplicity) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int it = first, nt = foundNtuplets->nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      continue;
    if (quality[it] == pixelTrack::Quality::edup)
      continue;
    assert(quality[it] == pixelTrack::Quality::bad);
    if (nhits > 7)
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits <= caConstants::maxHitsOnTrack);
    tupleMultiplicity->fill(nhits, it);
  }
}

__global__ void kernel_classifyTracks(HitContainer const *__restrict__ tuples,
                                      TkSoA const *__restrict__ tracks,
                                      CAHitNtupletGeneratorKernelsGPU::QualityCuts cuts,
                                      Quality *__restrict__ quality) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int it = first, nt = tuples->nOnes(); it < nt; it += gridDim.x * blockDim.x) {
    auto nhits = tuples->size(it);
    if (nhits == 0)
      break;  // guard

    // if duplicate: not even fit
    if (quality[it] == pixelTrack::Quality::edup)
      continue;

    assert(quality[it] == pixelTrack::Quality::bad);

    // mark doublets as bad
    if (nhits < 3)
      continue;

    // if the fit has any invalid parameters, mark it as bad
    bool isNaN = false;
    for (int i = 0; i < 5; ++i) {
      isNaN |= std::isnan(tracks->stateAtBS.state(it)(i));
    }
    if (isNaN) {
#ifdef NTUPLE_DEBUG
      printf("NaN in fit %d size %d chi2 %f\n", it, tuples->size(it), tracks->chi2(it));
#endif
      continue;
    }

    quality[it] = pixelTrack::Quality::strict;

    // compute a pT-dependent chi2 cut

    auto roughLog = [](float x) {
      // max diff [0.5,12] at 1.25 0.16143
      // average diff  0.0662998
      union IF {
        uint32_t i;
        float f;
      };
      IF z;
      z.f = x;
      uint32_t lsb = 1 < 21;
      z.i += lsb;
      z.i >>= 21;
      auto f = z.i & 3;
      int ex = int(z.i >> 2) - 127;

      // log2(1+0.25*f)
      // averaged over bins
      const float frac[4] = {0.160497f, 0.452172f, 0.694562f, 0.901964f};
      return float(ex) + frac[f];
    };

    // (see CAHitNtupletGeneratorGPU.cc)
    float pt = std::min<float>(tracks->pt(it), cuts.chi2MaxPt);
    float chi2Cut = cuts.chi2Scale * (cuts.chi2Coeff[0] + roughLog(pt) * cuts.chi2Coeff[1]);
    if (tracks->chi2(it) >= chi2Cut) {
#ifdef NTUPLE_FIT_DEBUG
      printf("Bad chi2 %d size %d pt %f eta %f chi2 %f\n",
             it,
             tuples->size(it),
             tracks->pt(it),
             tracks->eta(it),
             tracks->chi2(it));
#endif
      continue;
    }

    quality[it] = pixelTrack::Quality::tight;

    // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
    // default cuts:
    //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
    //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
    // (see CAHitNtupletGeneratorGPU.cc)
    auto const &region = (nhits > 3) ? cuts.quadruplet : cuts.triplet;
    bool isOk = (std::abs(tracks->tip(it)) < region.maxTip) and (tracks->pt(it) > region.minPt) and
                (std::abs(tracks->zip(it)) < region.maxZip);

    if (isOk)
      quality[it] = pixelTrack::Quality::highPurity;
  }
}

__global__ void kernel_doStatsForTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      break;  //guard
    if (quality[idx] < pixelTrack::Quality::loose)
      continue;
    atomicAdd(&(counters->nLooseTracks), 1);
    if (quality[idx] < pixelTrack::Quality::strict)
      continue;
    atomicAdd(&(counters->nGoodTracks), 1);
  }
}

__global__ void kernel_countHitInTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      break;  // guard
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->count(*h);
  }
}

__global__ void kernel_fillHitInTracks(HitContainer const *__restrict__ tuples,
                                       Quality const *__restrict__ quality,
                                       CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      break;  // guard
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fill(*h, idx);
  }
}

__global__ void kernel_fillHitDetIndices(HitContainer const *__restrict__ tuples,
                                         TrackingRecHit2DSOAView const *__restrict__ hhp,
                                         HitContainer *__restrict__ hitDetIndices) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  // copy offsets
  for (int idx = first, ntot = tuples->totOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    hitDetIndices->off[idx] = tuples->off[idx];
  }
  // fill hit indices
  auto const &hh = *hhp;
  auto nhits = hh.nHits();
  for (int idx = first, ntot = tuples->size(); idx < ntot; idx += gridDim.x * blockDim.x) {
    assert(tuples->content[idx] < nhits);
    hitDetIndices->content[idx] = hh.detectorIndex(tuples->content[idx]);
  }
}

__global__ void kernel_fillNLayers(TkSoA *__restrict__ ptracks, cms::cuda::AtomicPairCounter *apc) {
  auto &tracks = *ptracks;
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  auto ntracks = apc->get().m;
  if (0 == first)
    tracks.setNTracks(ntracks);
  for (int idx = first, nt = ntracks; idx < nt; idx += gridDim.x * blockDim.x) {
    auto nHits = tracks.nHits(idx);
    assert(nHits >= 3);
    tracks.nLayers(idx) = tracks.computeNumberOfLayers(idx);
  }
}

__global__ void kernel_doStatsForHitInTracks(CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ hitToTuple,
                                             CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto &c = *counters;
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple->size(idx) == 0)
      continue;  // SHALL NOT BE break
    atomicAdd(&c.nUsedHits, 1);
    if (hitToTuple->size(idx) > 1)
      atomicAdd(&c.nDupHits, 1);
  }
}

__global__ void kernel_countSharedHit(int *__restrict__ nshared,
                                      HitContainer const *__restrict__ ptuples,
                                      Quality const *__restrict__ quality,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  constexpr auto loose = pixelTrack::Quality::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &foundNtuplets = *ptuples;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    int nt = 0;

    // count "good" tracks
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (quality[*it] < loose)
        continue;
      ++nt;
    }

    if (nt < 2)
      continue;

    // now mark  each track triplet as sharing a hit
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (foundNtuplets.size(*it) > 3)
        continue;
      atomicAdd(&nshared[*it], 1);
    }

  }  //  hit loop
}

__global__ void kernel_markSharedHit(int const *__restrict__ nshared,
                                     HitContainer const *__restrict__ tuples,
                                     Quality *__restrict__ quality,
                                     bool dupPassThrough) {
  // constexpr auto bad = pixelTrack::Quality::bad;
  constexpr auto dup = pixelTrack::Quality::dup;
  constexpr auto loose = pixelTrack::Quality::loose;
  // constexpr auto strict = pixelTrack::Quality::strict;

  // quality to mark rejected
  auto const reject = dupPassThrough ? loose : dup;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (tuples->size(idx) == 0)
      break;  //guard
    if (quality[idx] <= reject)
      continue;
    if (nshared[idx] > 2)
      quality[idx] = reject;
  }
}

// mostly for very forward triplets.....
__global__ void kernel_rejectDuplicate(TkSoA const *__restrict__ ptracks,
                                       Quality *__restrict__ quality,
                                       uint16_t nmin,
                                       bool dupPassThrough,
                                       CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;

  auto &hitToTuple = *phitToTuple;
  auto const &tracks = *ptracks;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    /* chi2 is bad for large pt
    auto score = [&](auto it, auto nl) {
      return nl < 4 ? std::abs(tracks.tip(it)) :  // tip for triplets
                 tracks.chi2(it);                 //chi2
    };
    */
    auto score = [&](auto it, auto nl) { return std::abs(tracks.tip(it)); };

    // full combinatorics
    for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
      auto const it = *ip;
      auto qi = quality[it];
      if (qi <= reject)
        continue;
      auto opi = tracks.stateAtBS.state(it)(2);
      auto e2opi = tracks.stateAtBS.covariance(it)(9);
      auto cti = tracks.stateAtBS.state(it)(3);
      auto e2cti = tracks.stateAtBS.covariance(it)(12);
      auto nli = tracks.nLayers(it);
      for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
        auto const jt = *jp;
        auto qj = quality[jt];
        if (qj <= reject)
          continue;
        auto opj = tracks.stateAtBS.state(jt)(2);
        auto ctj = tracks.stateAtBS.state(jt)(3);
        auto dct = nSigma2 * (tracks.stateAtBS.covariance(jt)(12) + e2cti);
        if ((cti - ctj) * (cti - ctj) > dct)
          continue;
        auto dop = nSigma2 * (tracks.stateAtBS.covariance(jt)(9) + e2opi);
        if ((opi - opj) * (opi - opj) > dop)
          continue;
        auto nlj = tracks.nLayers(jt);
        if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
          quality[jt] = reject;
        else {
          quality[it] = reject;
          break;
        }
      }
    }
  }
}

__global__ void kernel_sharedHitCleaner(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                        TkSoA const *__restrict__ ptracks,
                                        Quality *__restrict__ quality,
                                        int nmin,
                                        bool dupPassThrough,
                                        CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = dupPassThrough ? pixelTrack::Quality::loose : pixelTrack::Quality::dup;
  // quality of longest track
  auto const longTqual = pixelTrack::Quality::highPurity;

  auto &hitToTuple = *phitToTuple;
  auto const &tracks = *ptracks;

  auto const &hh = *hhp;
  int l1end = hh.hitsLayerStart()[1];

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    int8_t maxNl = 0;

    // find maxNl
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (quality[*it] < longTqual)
        continue;
      // if (tracks.nHits(*it)==3) continue;
      auto nl = tracks.nLayers(*it);
      maxNl = std::max(nl, maxNl);
    }

    if (maxNl < 4)
      continue;

    // quad pass through (leave for tests)
    // maxNl = std::min(4, maxNl);

    // kill all tracks shorter than maxHl (only triplets???
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      auto nl = tracks.nLayers(*it);

      //checking if shared hit is on bpix1 and if the tuple is short enough
      if (idx < l1end and nl > nmin)
        continue;

      if (nl < maxNl && quality[*it] > reject)
        quality[*it] = reject;
    }
  }
}

__global__ void kernel_tripletCleaner(TkSoA const *__restrict__ ptracks,
                                      Quality *__restrict__ quality,
                                      uint16_t nmin,
                                      bool dupPassThrough,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = pixelTrack::Quality::loose;
  /// min quality of good
  auto const good = pixelTrack::Quality::strict;

  auto &hitToTuple = *phitToTuple;
  auto const &tracks = *ptracks;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;
    bool onlyTriplets = true;

    // check if only triplets
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      if (quality[*it] <= good)
        continue;
      onlyTriplets &= tracks.isTriplet(*it);
      if (!onlyTriplets)
        break;
    }

    // only triplets
    if (!onlyTriplets)
      continue;

    // for triplets choose best tip!  (should we first find best quality???)
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] >= good && std::abs(tracks.tip(it)) < mc) {
        mc = std::abs(tracks.tip(it));
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark worse ambiguities
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] > reject && it != im)
        quality[it] = reject;  //no race:  simple assignment of the same constant
    }

  }  // loop over hits
}

__global__ void kernel_simpleTripletCleaner(
    TkSoA const *__restrict__ ptracks,
    Quality *__restrict__ quality,
    uint16_t nmin,
    bool dupPassThrough,
    CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  // quality to mark rejected
  auto const reject = pixelTrack::Quality::loose;
  /// min quality of good
  auto const good = pixelTrack::Quality::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &tracks = *ptracks;

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple.nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = maxScore;
    uint16_t im = tkNotFound;

    // choose best tip!  (should we first find best quality???)
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] >= good && std::abs(tracks.tip(it)) < mc) {
        mc = std::abs(tracks.tip(it));
        im = it;
      }
    }

    if (tkNotFound == im)
      continue;

    // mark worse ambiguities
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] > reject && tracks.isTriplet(it) && it != im)
        quality[it] = reject;  //no race:  simple assignment of the same constant
    }

  }  // loop over hits
}

__global__ void kernel_print_found_ntuplets(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                            HitContainer const *__restrict__ ptuples,
                                            TkSoA const *__restrict__ ptracks,
                                            Quality const *__restrict__ quality,
                                            CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple,
                                            int32_t firstPrint,
                                            int32_t lastPrint,
                                            int iev) {
  constexpr auto loose = pixelTrack::Quality::loose;
  auto const &hh = *hhp;
  auto const &foundNtuplets = *ptuples;
  auto const &tracks = *ptracks;
  int first = firstPrint + blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first, np = std::min(lastPrint, foundNtuplets.nOnes()); i < np; i += blockDim.x * gridDim.x) {
    auto nh = foundNtuplets.size(i);
    if (nh < 3)
      continue;
    if (quality[i] < loose)
      continue;
    printf("TK: %d %d %d %d %f %f %f %f %f %f %f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
           10000 * iev + i,
           int(quality[i]),
           nh,
           tracks.nLayers(i),
           tracks.charge(i),
           tracks.pt(i),
           tracks.eta(i),
           tracks.phi(i),
           tracks.tip(i),
           tracks.zip(i),
           //           asinhf(fit_results[i].par(3)),
           tracks.chi2(i),
           hh.zGlobal(*foundNtuplets.begin(i)),
           hh.zGlobal(*(foundNtuplets.begin(i) + 1)),
           hh.zGlobal(*(foundNtuplets.begin(i) + 2)),
           nh > 3 ? hh.zGlobal(int(*(foundNtuplets.begin(i) + 3))) : 0,
           nh > 4 ? hh.zGlobal(int(*(foundNtuplets.begin(i) + 4))) : 0,
           nh > 5 ? hh.zGlobal(int(*(foundNtuplets.begin(i) + 5))) : 0,
           nh > 6 ? hh.zGlobal(int(*(foundNtuplets.begin(i) + nh - 1))) : 0);
  }
}

__global__ void kernel_printCounters(cAHitNtupletGenerator::Counters const *counters) {
  auto const &c = *counters;
  printf(
      "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
      "nDupHits | "
      "nFishCells | "
      "nKilledCells | "
      "nUsedCells | nZeroTrackCells ||\n");
  printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
         c.nEvents,
         c.nHits,
         c.nCells,
         c.nTuples,
         c.nFitTracks,
         c.nLooseTracks,
         c.nGoodTracks,
         c.nUsedHits,
         c.nDupHits,
         c.nFishCells,
         c.nKilledCells,
         c.nEmptyCells,
         c.nZeroTrackCells);
  printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  %.3f||\n",
         c.nEvents,
         c.nHits / double(c.nEvents),
         c.nCells / double(c.nEvents),
         c.nTuples / double(c.nEvents),
         c.nFitTracks / double(c.nEvents),
         c.nLooseTracks / double(c.nEvents),
         c.nGoodTracks / double(c.nEvents),
         c.nUsedHits / double(c.nEvents),
         c.nDupHits / double(c.nEvents),
         c.nFishCells / double(c.nCells),
         c.nKilledCells / double(c.nCells),
         c.nEmptyCells / double(c.nCells),
         c.nZeroTrackCells / double(c.nCells));
}
