#ifndef RecoPixelVertexing_PixelTriplets_plugins_GPUCACellT_h
#define RecoPixelVertexing_PixelTriplets_plugins_GPUCACellT_h

//
// Author: Felice Pantaleo, CERN
//

// #define ONLY_TRIPLETS_IN_HOLE

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CircleEq.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "CAStructures.h"

template <typename TrackerTraits>
class GPUCACellT {
public:
  using PtrAsInt = unsigned long long;

  static constexpr auto maxCellsPerHit = TrackerTraits::maxCellsPerHit;
  using OuterHitOfCellContainer = caStructures::OuterHitOfCellContainerT<TrackerTraits>;
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

  using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
  using hindex_type = typename TrackerTraits::hindex_type;
  using tindex_type = typename TrackerTraits::tindex_type;
  static constexpr auto invalidHitId = std::numeric_limits<hindex_type>::max();

  using TmpTuple = cms::cuda::VecArray<uint32_t, TrackerTraits::maxDepth>;

  using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;
  using Quality = pixelTrack::Quality;
  static constexpr auto bad = pixelTrack::Quality::bad;

  enum class StatusBit : uint16_t { kUsed = 1, kInTrack = 2, kKilled = 1 << 15 };

  GPUCACellT() = default;

  __device__ __forceinline__ void init(CellNeighborsVector& cellNeighbors,
                                       CellTracksVector& cellTracks,
                                       const HitsConstView& hh,
                                       int layerPairId,
                                       hindex_type innerHitId,
                                       hindex_type outerHitId) {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theLayerPairId_ = layerPairId;
    theStatus_ = 0;
    theFishboneId = invalidHitId;

    // optimization that depends on access pattern
    theInnerZ = hh[innerHitId].zGlobal();
    theInnerR = hh[innerHitId].rGlobal();

    // link to default empty
    theOuterNeighbors = &cellNeighbors[0];
    theTracks = &cellTracks[0];
    assert(outerNeighbors().empty());
    assert(tracks().empty());
  }

  __device__ __forceinline__ int addOuterNeighbor(typename TrackerTraits::cindex_type t,
                                                  CellNeighborsVector& cellNeighbors) {
    // use smart cache
    if (outerNeighbors().empty()) {
      auto i = cellNeighbors.extend();  // maybe wasted....
      if (i > 0) {
        cellNeighbors[i].reset();
        __threadfence();
#ifdef __CUDACC__
        auto zero = (PtrAsInt)(&cellNeighbors[0]);
        atomicCAS((PtrAsInt*)(&theOuterNeighbors),
                  zero,
                  (PtrAsInt)(&cellNeighbors[i]));  // if fails we cannot give "i" back...
#else
        theOuterNeighbors = &cellNeighbors[i];
#endif
      } else
        return -1;
    }
    __threadfence();
    return outerNeighbors().push_back(t);
  }

  __device__ __forceinline__ int addTrack(tindex_type t, CellTracksVector& cellTracks) {
    if (tracks().empty()) {
      auto i = cellTracks.extend();  // maybe wasted....
      if (i > 0) {
        cellTracks[i].reset();
        __threadfence();
#ifdef __CUDACC__
        auto zero = (PtrAsInt)(&cellTracks[0]);
        atomicCAS((PtrAsInt*)(&theTracks), zero, (PtrAsInt)(&cellTracks[i]));  // if fails we cannot give "i" back...
#else
        theTracks = &cellTracks[i];
#endif
      } else
        return -1;
    }
    __threadfence();
    return tracks().push_back(t);
  }

  __device__ __forceinline__ CellTracks& tracks() { return *theTracks; }
  __device__ __forceinline__ CellTracks const& tracks() const { return *theTracks; }
  __device__ __forceinline__ CellNeighbors& outerNeighbors() { return *theOuterNeighbors; }
  __device__ __forceinline__ CellNeighbors const& outerNeighbors() const { return *theOuterNeighbors; }
  __device__ __forceinline__ float inner_x(const HitsConstView& hh) const { return hh[theInnerHitId].xGlobal(); }
  __device__ __forceinline__ float outer_x(const HitsConstView& hh) const { return hh[theOuterHitId].xGlobal(); }
  __device__ __forceinline__ float inner_y(const HitsConstView& hh) const { return hh[theInnerHitId].yGlobal(); }
  __device__ __forceinline__ float outer_y(const HitsConstView& hh) const { return hh[theOuterHitId].yGlobal(); }
  __device__ __forceinline__ float inner_z(const HitsConstView& hh) const { return theInnerZ; }
  // { return hh.zGlobal(theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float outer_z(const HitsConstView& hh) const { return hh[theOuterHitId].zGlobal(); }
  __device__ __forceinline__ float inner_r(const HitsConstView& hh) const { return theInnerR; }
  // { return hh.rGlobal(theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float outer_r(const HitsConstView& hh) const { return hh[theOuterHitId].rGlobal(); }

  __device__ __forceinline__ auto inner_iphi(const HitsConstView& hh) const { return hh[theInnerHitId].iphi(); }
  __device__ __forceinline__ auto outer_iphi(const HitsConstView& hh) const { return hh[theOuterHitId].iphi(); }

  __device__ __forceinline__ float inner_detIndex(const HitsConstView& hh) const {
    return hh[theInnerHitId].detectorIndex();
  }
  __device__ __forceinline__ float outer_detIndex(const HitsConstView& hh) const {
    return hh[theOuterHitId].detectorIndex();
  }

  constexpr unsigned int inner_hit_id() const { return theInnerHitId; }
  constexpr unsigned int outer_hit_id() const { return theOuterHitId; }

  __device__ void print_cell() const {
    printf("printing cell: on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
           theLayerPairId_,
           theInnerHitId,
           theOuterHitId);
  }

  __device__ bool check_alignment(const HitsConstView& hh,
                                  GPUCACellT const& otherCell,
                                  const float ptmin,
                                  const float hardCurvCut,
                                  const float caThetaCutBarrel,
                                  const float caThetaCutForward,
                                  const float dcaCutInnerTriplet,
                                  const float dcaCutOuterTriplet) const {
    // detIndex of the layerStart for the Phase1 Pixel Detector:
    // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
    // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
    auto ri = inner_r(hh);
    auto zi = inner_z(hh);

    auto ro = outer_r(hh);
    auto zo = outer_z(hh);

    auto r1 = otherCell.inner_r(hh);
    auto z1 = otherCell.inner_z(hh);
    auto isBarrel = otherCell.outer_detIndex(hh) < TrackerTraits::last_barrel_detIndex;
    bool aligned = areAlignedRZ(r1,
                                z1,
                                ri,
                                zi,
                                ro,
                                zo,
                                ptmin,
                                isBarrel ? caThetaCutBarrel : caThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
    return (aligned && dcaCut(hh,
                              otherCell,
                              otherCell.inner_detIndex(hh) < TrackerTraits::last_bpix1_detIndex ? dcaCutInnerTriplet
                                                                                                : dcaCutOuterTriplet,
                              hardCurvCut));  // FIXME tune cuts
  }

  __device__ __forceinline__ static bool areAlignedRZ(
      float r1, float z1, float ri, float zi, float ro, float zo, const float ptmin, const float thetaCut) {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    float pMin = ptmin * std::sqrt(distance_13_squared);  // this needs to be divided by
                                                          // radius_diff later

    float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
    return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
  }

  __device__ inline bool dcaCut(const HitsConstView& hh,
                                GPUCACellT const& otherCell,
                                const float region_origin_radius_plus_tolerance,
                                const float maxCurv) const {
    auto x1 = otherCell.inner_x(hh);
    auto y1 = otherCell.inner_y(hh);

    auto x2 = inner_x(hh);
    auto y2 = inner_y(hh);

    auto x3 = outer_x(hh);
    auto y3 = outer_y(hh);

    CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

    if (eq.curvature() > maxCurv)
      return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
  }

  __device__ __forceinline__ static bool dcaCutH(float x1,
                                                 float y1,
                                                 float x2,
                                                 float y2,
                                                 float x3,
                                                 float y3,
                                                 const float region_origin_radius_plus_tolerance,
                                                 const float maxCurv) {
    CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

    if (eq.curvature() > maxCurv)
      return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
  }

  __device__ inline bool hole0(const HitsConstView& hh, GPUCACellT const& innerCell) const {
    using namespace phase1PixelTopology;

    int p = innerCell.inner_iphi(hh);
    if (p < 0)
      p += std::numeric_limits<unsigned short>::max();
    p = (max_ladder_bpx0 * p) / std::numeric_limits<unsigned short>::max();
    p %= max_ladder_bpx0;
    auto il = first_ladder_bpx0 + p;
    auto r0 = hh.averageGeometry().ladderR[il];
    auto ri = innerCell.inner_r(hh);
    auto zi = innerCell.inner_z(hh);
    auto ro = outer_r(hh);
    auto zo = outer_z(hh);
    auto z0 = zi + (r0 - ri) * (zo - zi) / (ro - ri);
    auto z_in_ladder = std::abs(z0 - hh.averageGeometry().ladderZ[il]);
    auto z_in_module = z_in_ladder - module_length_bpx0 * int(z_in_ladder / module_length_bpx0);
    auto gap = z_in_module < module_tolerance_bpx0 || z_in_module > (module_length_bpx0 - module_tolerance_bpx0);
    return gap;
  }

  __device__ inline bool hole4(const HitsConstView& hh, GPUCACellT const& innerCell) const {
    using namespace phase1PixelTopology;

    int p = outer_iphi(hh);
    if (p < 0)
      p += std::numeric_limits<unsigned short>::max();
    p = (max_ladder_bpx4 * p) / std::numeric_limits<unsigned short>::max();
    p %= max_ladder_bpx4;
    auto il = first_ladder_bpx4 + p;
    auto r4 = hh.averageGeometry().ladderR[il];
    auto ri = innerCell.inner_r(hh);
    auto zi = innerCell.inner_z(hh);
    auto ro = outer_r(hh);
    auto zo = outer_z(hh);
    auto z4 = zo + (r4 - ro) * (zo - zi) / (ro - ri);
    auto z_in_ladder = std::abs(z4 - hh.averageGeometry().ladderZ[il]);
    auto z_in_module = z_in_ladder - module_length_bpx4 * int(z_in_ladder / module_length_bpx4);
    auto gap = z_in_module < module_tolerance_bpx4 || z_in_module > (module_length_bpx4 - module_tolerance_bpx4);
    auto holeP = z4 > hh.averageGeometry().ladderMaxZ[il] && z4 < hh.averageGeometry().endCapZ[0];
    auto holeN = z4 < hh.averageGeometry().ladderMinZ[il] && z4 > hh.averageGeometry().endCapZ[1];
    return gap || holeP || holeN;
  }

  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.

  template <int DEPTH>
  __device__ inline void find_ntuplets(const HitsConstView& hh,
                                       GPUCACellT* __restrict__ cells,
                                       CellTracksVector& cellTracks,
                                       HitContainer& foundNtuplets,
                                       cms::cuda::AtomicPairCounter& apc,
                                       Quality* __restrict__ quality,
                                       TmpTuple& tmpNtuplet,
                                       const unsigned int minHitsPerNtuplet,
                                       bool startAt0) const {
    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold

    if constexpr (DEPTH <= 0) {
      printf("ERROR: GPUCACellT::find_ntuplets reached full depth!\n");
#ifdef __CUDA_ARCH__
      __trap();
#else
      abort();
#endif
    } else {
      auto doubletId = this - cells;
      tmpNtuplet.push_back_unsafe(doubletId);
      assert(tmpNtuplet.size() <=
             int(TrackerTraits::maxHitsOnTrack -
                 3));  //1 for the container, 1 because these are doublets, 1 because we may push another

      bool last = true;
      for (unsigned int otherCell : outerNeighbors()) {
        if (cells[otherCell].isKilled())
          continue;  // killed by earlyFishbone
        last = false;
        cells[otherCell].template find_ntuplets<DEPTH - 1>(
            hh, cells, cellTracks, foundNtuplets, apc, quality, tmpNtuplet, minHitsPerNtuplet, startAt0);
      }

      if (last) {  // if long enough save...
        if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifdef ONLY_TRIPLETS_IN_HOLE
          // triplets accepted only pointing to the hole
          if (tmpNtuplet.size() >= 3 || (startAt0 && hole4(hh, cells[tmpNtuplet[0]])) ||
              ((!startAt0) && hole0(hh, cells[tmpNtuplet[0]])))
#endif
          {
            hindex_type hits[TrackerTraits::maxDepth + 2];
            auto nh = 0U;
            constexpr int maxFB = 2;  // for the time being let's limit this
            int nfb = 0;
            for (auto c : tmpNtuplet) {
              hits[nh++] = cells[c].theInnerHitId;
              if (nfb < maxFB && cells[c].hasFishbone()) {
                ++nfb;
                hits[nh++] = cells[c].theFishboneId;  // fishbone hit is always outer than inner hit
              }
            }
            assert(nh < TrackerTraits::maxHitsOnTrack);
            hits[nh] = theOuterHitId;
            auto it = foundNtuplets.bulkFill(apc, hits, nh + 1);
            if (it >= 0) {  // if negative is overflow....
              for (auto c : tmpNtuplet)
                cells[c].addTrack(it, cellTracks);
              quality[it] = bad;  // initialize to bad
            }
          }
        }
      }
      tmpNtuplet.pop_back();
      assert(tmpNtuplet.size() < int(TrackerTraits::maxHitsOnTrack - 1));
    }
  }

  // Cell status management
  __device__ __forceinline__ void kill() { theStatus_ |= uint16_t(StatusBit::kKilled); }
  __device__ __forceinline__ bool isKilled() const { return theStatus_ & uint16_t(StatusBit::kKilled); }

  __device__ __forceinline__ int16_t layerPairId() const { return theLayerPairId_; }

  __device__ __forceinline__ bool unused() const { return 0 == (uint16_t(StatusBit::kUsed) & theStatus_); }
  __device__ __forceinline__ void setStatusBits(StatusBit mask) { theStatus_ |= uint16_t(mask); }

  __device__ __forceinline__ void setFishbone(hindex_type id, float z, const HitsConstView& hh) {
    // make it deterministic: use the farther apart (in z)
    auto old = theFishboneId;
    while (old !=
           atomicCAS(
               &theFishboneId,
               old,
               (invalidHitId == old || std::abs(z - theInnerZ) > std::abs(hh[old].zGlobal() - theInnerZ)) ? id : old))
      old = theFishboneId;
  }
  __device__ __forceinline__ auto fishboneId() const { return theFishboneId; }
  __device__ __forceinline__ bool hasFishbone() const { return theFishboneId != invalidHitId; }

private:
  CellNeighbors* theOuterNeighbors;
  CellTracks* theTracks;

  int16_t theLayerPairId_;
  uint16_t theStatus_;  // tbd

  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
  hindex_type theFishboneId;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_GPUCACellT_h
