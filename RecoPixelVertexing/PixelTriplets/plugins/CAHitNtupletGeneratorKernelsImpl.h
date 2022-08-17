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

#include "CAStructures.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

namespace caHitNtupletGeneratorKernels {

  constexpr uint32_t tkNotFound = std::numeric_limits<uint16_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

  //all of these below are mostly to avoid brining around the relative namespace
  template <typename TrackerTraits>
  using HitsView = TrackingRecHit2DSOAViewT<TrackerTraits>;

  template <typename TrackerTraits>
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;

  template <typename TrackerTraits>
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  template <typename TrackerTraits>
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;

  template <typename TrackerTraits>
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

  template <typename TrackerTraits>
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using Quality = pixelTrack::Quality;

  template <typename TrackerTraits>
  using TkSoA = pixelTrack::TrackSoAT<TrackerTraits>;

  template <typename TrackerTraits>
  using HitContainer = pixelTrack::HitContainerT<TrackerTraits>;

  template <typename TrackerTraits>
  using Hits = typename GPUCACellT<TrackerTraits>::Hits;

  template <typename TrackerTraits>
  using QualityCuts = pixelTrack::QualityCutsT<TrackerTraits>;

  template <typename TrackerTraits>
  using CAParams = caHitNtupletGenerator::CAParamsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;

  template <typename TrackerTraits>
  __global__ void kernel_checkOverflows(HitContainer<TrackerTraits> const *foundNtuplets,
                                        TupleMultiplicity<TrackerTraits> const *tupleMultiplicity,
                                        HitToTuple<TrackerTraits> const *hitToTuple,
                                        cms::cuda::AtomicPairCounter *apc,
                                        GPUCACellT<TrackerTraits> const *__restrict__ cells,
                                        uint32_t const *__restrict__ nCells,
                                        CellNeighborsVector<TrackerTraits> const *cellNeighbors,
                                        CellTracksVector<TrackerTraits> const *cellTracks,
                                        OuterHitOfCell<TrackerTraits> const isOuterHitOfCell,
                                        int32_t nHits,
                                        uint32_t maxNumberOfDoublets,
                                        Counters *counters) {
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
      printf("number of found cells %d \n found tuples %d with total hits %d out of %d %d\n",
             *nCells,
             apc->get().m,
             apc->get().n,
             nHits,
             hitToTuple->totOnes());
      if (apc->get().m < TrackerTraits::maxNumberOfQuadruplets) {
        assert(foundNtuplets->size(apc->get().m) == 0);
        assert(foundNtuplets->size() == apc->get().n);
      }
    }

    for (int idx = first, nt = foundNtuplets->nOnes(); idx < nt; idx += gridDim.x * blockDim.x) {
      if (foundNtuplets->size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
        printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
      assert(foundNtuplets->size(idx) <= TrackerTraits::maxHitsOnTrack);
      for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
        assert(int(*ih) < nHits);
    }
#endif

    if (0 == first) {
      if (apc->get().m >= TrackerTraits::maxNumberOfQuadruplets)
        printf("Tuples overflow\n");
      if (*nCells >= maxNumberOfDoublets)
        printf("Cells overflow\n");
      if (cellNeighbors && cellNeighbors->full())
        printf("cellNeighbors overflow %d %d \n", cellNeighbors->capacity(), cellNeighbors->size());
      if (cellTracks && cellTracks->full())
        printf("cellTracks overflow\n");
      if (int(hitToTuple->nOnes()) < nHits)
        printf("ERROR hitToTuple  overflow %d %d\n", hitToTuple->nOnes(), nHits);
#ifdef GPU_DEBUG
      printf("size of cellNeighbors %d \n cellTracks %d \n hitToTuple %d \n",
             cellNeighbors->size(),
             cellTracks->size(),
             hitToTuple->size());
// printf("cellTracksSizes;");
// for (int i = 0; i < cellTracks->size(); i++) {
//   printf("%d;",cellTracks[i].size());
// }
//
// printf("\n");
// printf("cellNeighborsSizes;");
// for (int i = 0; i < cellNeighbors->size(); i++) {
//   printf("%d;",cellNeighbors[i].size());
// }
// printf("\n");
#endif
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

  template <typename TrackerTraits>
  __global__ void kernel_fishboneCleaner(GPUCACellT<TrackerTraits> const *cells,
                                         uint32_t const *__restrict__ nCells,
                                         Quality *quality) {
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
  template <typename TrackerTraits>
  __global__ void kernel_earlyDuplicateRemover(GPUCACellT<TrackerTraits> const *cells,
                                               uint32_t const *__restrict__ nCells,
                                               TkSoA<TrackerTraits> const *__restrict__ ptracks,
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
  template <typename TrackerTraits>
  __global__ void kernel_fastDuplicateRemover(GPUCACellT<TrackerTraits> const *__restrict__ cells,
                                              uint32_t const *__restrict__ nCells,
                                              TkSoA<TrackerTraits> *__restrict__ tracks,
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

  template <typename TrackerTraits>
  __global__ void kernel_connect(cms::cuda::AtomicPairCounter *apc1,
                                 cms::cuda::AtomicPairCounter *apc2,  // just to zero them,
                                 Hits<TrackerTraits> const *__restrict__ hhp,
                                 GPUCACellT<TrackerTraits> *cells,
                                 uint32_t const *__restrict__ nCells,
                                 CellNeighborsVector<TrackerTraits> *cellNeighbors,
                                 OuterHitOfCell<TrackerTraits> const isOuterHitOfCell,
                                 CAParams<TrackerTraits> params) {
    using Cell = GPUCACellT<TrackerTraits>;
    auto const &hh = *hhp;

    auto firstCellIndex = threadIdx.y + blockIdx.y * blockDim.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    if (0 == (firstCellIndex + first)) {
      (*apc1) = 0;
      (*apc2) = 0;
    }  // ready for next kernel

    constexpr uint32_t last_bpix1_detIndex = TrackerTraits::last_bpix1_detIndex;
    constexpr uint32_t last_barrel_detIndex = TrackerTraits::last_barrel_detIndex;
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
      auto isBarrel = thisCell.inner_detIndex(hh) < last_barrel_detIndex;

      for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
        auto otherCell = __ldg(vi + j);
        auto &oc = cells[otherCell];
        auto r1 = oc.inner_r(hh);
        auto z1 = oc.inner_z(hh);
        bool aligned = Cell::areAlignedRZ(
            r1,
            z1,
            ri,
            zi,
            ro,
            zo,
            params.ptmin_,
            isBarrel ? params.CAThetaCutBarrel_ : params.CAThetaCutForward_);  // 2.f*thetaCut); // FIXME tune cuts
        if (aligned && thisCell.dcaCut(hh,
                                       oc,
                                       oc.inner_detIndex(hh) < last_bpix1_detIndex ? params.dcaCutInnerTriplet_
                                                                                   : params.dcaCutOuterTriplet_,
                                       params.hardCurvCut_)) {  // FIXME tune cuts
          oc.addOuterNeighbor(cellIndex, *cellNeighbors);
          thisCell.setStatusBits(Cell::StatusBit::kUsed);
          oc.setStatusBits(Cell::StatusBit::kUsed);
        }
      }  // loop on inner cells
    }    // loop on outer cells
  }

  template <typename TrackerTraits>
  __global__ void kernel_find_ntuplets(Hits<TrackerTraits> const *__restrict__ hhp,
                                       GPUCACellT<TrackerTraits> *__restrict__ cells,
                                       uint32_t const *nCells,
                                       CellTracksVector<TrackerTraits> *cellTracks,
                                       HitContainer<TrackerTraits> *foundNtuplets,
                                       cms::cuda::AtomicPairCounter *apc,
                                       Quality *__restrict__ quality,
                                       CAParams<TrackerTraits> params) {
    // recursive: not obvious to widen
    auto const &hh = *hhp;

    using Cell = GPUCACellT<TrackerTraits>;

    auto first = threadIdx.x + blockIdx.x * blockDim.x;

#ifdef GPU_DEBUG
    if (first == 0)
      printf("starting producing ntuplets from %d cells \n", *nCells);
#endif
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto const &thisCell = cells[idx];

      if (thisCell.isKilled())
        continue;  // cut by earlyFishbone

      // we require at least three hits...
      if (thisCell.outerNeighbors().empty())
        continue;

      auto pid = thisCell.layerPairId();
      bool doit = params.startingLayerPair(pid);

      constexpr uint32_t maxDepth = TrackerTraits::maxDepth;
      if (doit) {
        typename Cell::TmpTuple stack;
        stack.reset();

        bool bpix1Start = params.startAt0(pid);

        thisCell.template find_ntuplets<maxDepth>(
            hh, cells, *cellTracks, *foundNtuplets, *apc, quality, stack, params.minHitsPerNtuplet_, bpix1Start);

        assert(stack.empty());
      }
    }
  }
  template <typename TrackerTraits>
  __global__ void kernel_mark_used(GPUCACellT<TrackerTraits> *__restrict__ cells, uint32_t const *nCells) {
    auto first = threadIdx.x + blockIdx.x * blockDim.x;
    using Cell = GPUCACellT<TrackerTraits>;
    for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      auto &thisCell = cells[idx];
      if (!thisCell.tracks().empty())
        thisCell.setStatusBits(Cell::StatusBit::kInTrack);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_countMultiplicity(HitContainer<TrackerTraits> const *__restrict__ foundNtuplets,
                                           Quality const *__restrict__ quality,
                                           TupleMultiplicity<TrackerTraits> *tupleMultiplicity) {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = first, nt = foundNtuplets->nOnes(); it < nt; it += gridDim.x * blockDim.x) {
      auto nhits = foundNtuplets->size(it);
      if (nhits < 3)
        continue;
      if (quality[it] == pixelTrack::Quality::edup)
        continue;
      assert(quality[it] == pixelTrack::Quality::bad);
      if (nhits > TrackerTraits::maxHitsOnTrack)  // current limit
        printf("wrong mult %d %d\n", it, nhits);
      assert(nhits <= TrackerTraits::maxHitsOnTrack);
      tupleMultiplicity->count(nhits);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillMultiplicity(HitContainer<TrackerTraits> const *__restrict__ foundNtuplets,
                                          Quality const *__restrict__ quality,
                                          TupleMultiplicity<TrackerTraits> *tupleMultiplicity) {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = first, nt = foundNtuplets->nOnes(); it < nt; it += gridDim.x * blockDim.x) {
      auto nhits = foundNtuplets->size(it);
      if (nhits < 3)
        continue;
      if (quality[it] == pixelTrack::Quality::edup)
        continue;
      assert(quality[it] == pixelTrack::Quality::bad);
      if (nhits > TrackerTraits::maxHitsOnTrack)
        printf("wrong mult %d %d\n", it, nhits);
      assert(nhits <= TrackerTraits::maxHitsOnTrack);
      tupleMultiplicity->fill(nhits, it);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_classifyTracks(HitContainer<TrackerTraits> const *__restrict__ tuples,
                                        TkSoA<TrackerTraits> const *__restrict__ tracks,
                                        QualityCuts<TrackerTraits> cuts,
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

      if (cuts.strictCut(tracks, it))
        continue;

      quality[it] = pixelTrack::Quality::tight;

      if (cuts.isHP(tracks, nhits, it))
        quality[it] = pixelTrack::Quality::highPurity;
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_doStatsForTracks(HitContainer<TrackerTraits> const *__restrict__ tuples,
                                          Quality const *__restrict__ quality,
                                          Counters *counters) {
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

  template <typename TrackerTraits>
  __global__ void kernel_countHitInTracks(HitContainer<TrackerTraits> const *__restrict__ tuples,
                                          Quality const *__restrict__ quality,
                                          HitToTuple<TrackerTraits> *hitToTuple) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tuples->size(idx) == 0)
        break;  // guard
      for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
        hitToTuple->count(*h);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillHitInTracks(HitContainer<TrackerTraits> const *__restrict__ tuples,
                                         Quality const *__restrict__ quality,
                                         HitToTuple<TrackerTraits> *hitToTuple) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = first, ntot = tuples->nOnes(); idx < ntot; idx += gridDim.x * blockDim.x) {
      if (tuples->size(idx) == 0)
        break;  // guard
      for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
        hitToTuple->fill(*h, idx);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_fillHitDetIndices(HitContainer<TrackerTraits> const *__restrict__ tuples,
                                           HitsView<TrackerTraits> const *__restrict__ hhp,
                                           HitContainer<TrackerTraits> *__restrict__ hitDetIndices) {
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

  template <typename TrackerTraits>
  __global__ void kernel_fillNLayers(TkSoA<TrackerTraits> *__restrict__ ptracks, cms::cuda::AtomicPairCounter *apc) {
    auto &tracks = *ptracks;
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    // clamp the number of tracks to the capacity of the SoA
    auto ntracks = std::min<int>(apc->get().m, tracks.stride() - 1);
    if (0 == first)
      tracks.setNTracks(ntracks);
    for (int idx = first, nt = ntracks; idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      assert(nHits >= 3);
      tracks.nLayers(idx) = tracks.computeNumberOfLayers(idx);
    }
  }

  template <typename TrackerTraits>
  __global__ void kernel_doStatsForHitInTracks(HitToTuple<TrackerTraits> const *__restrict__ hitToTuple,
                                               Counters *counters) {
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

  template <typename TrackerTraits>
  __global__ void kernel_countSharedHit(int *__restrict__ nshared,
                                        HitContainer<TrackerTraits> const *__restrict__ ptuples,
                                        Quality const *__restrict__ quality,
                                        HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
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

  template <typename TrackerTraits>
  __global__ void kernel_markSharedHit(int const *__restrict__ nshared,
                                       HitContainer<TrackerTraits> const *__restrict__ tuples,
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
  template <typename TrackerTraits>
  __global__ void kernel_rejectDuplicate(TkSoA<TrackerTraits> const *__restrict__ ptracks,
                                         Quality *__restrict__ quality,
                                         uint16_t nmin,
                                         bool dupPassThrough,
                                         HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
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

  template <typename TrackerTraits>
  __global__ void kernel_sharedHitCleaner(HitsView<TrackerTraits> const *__restrict__ hhp,
                                          TkSoA<TrackerTraits> const *__restrict__ ptracks,
                                          Quality *__restrict__ quality,
                                          int nmin,
                                          bool dupPassThrough,
                                          HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
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

  template <typename TrackerTraits>
  __global__ void kernel_tripletCleaner(TkSoA<TrackerTraits> const *__restrict__ ptracks,
                                        Quality *__restrict__ quality,
                                        uint16_t nmin,
                                        bool dupPassThrough,
                                        HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
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

  template <typename TrackerTraits>
  __global__ void kernel_simpleTripletCleaner(TkSoA<TrackerTraits> const *__restrict__ ptracks,
                                              Quality *__restrict__ quality,
                                              uint16_t nmin,
                                              bool dupPassThrough,
                                              HitToTuple<TrackerTraits> const *__restrict__ phitToTuple) {
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

  template <typename TrackerTraits>
  __global__ void kernel_print_found_ntuplets(HitsView<TrackerTraits> const *__restrict__ hhp,
                                              HitContainer<TrackerTraits> const *__restrict__ ptuples,
                                              TkSoA<TrackerTraits> const *__restrict__ ptracks,
                                              Quality const *__restrict__ quality,
                                              HitToTuple<TrackerTraits> const *__restrict__ phitToTuple,
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

  __global__ void kernel_printCounters(Counters const *counters) {
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
    printf(
        "Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  %.3f||\n",
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

}  // namespace caHitNtupletGeneratorKernels
