#ifndef RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
#define RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h

//
// Author: Felice Pantaleo, CERN
//

// #define ONLY_TRIPLETS_IN_HOLE

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CircleEq.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CAConstants.h"

class GPUCACell {
public:
  using PtrAsInt = unsigned long long;

  static constexpr auto maxCellsPerHit = caConstants::maxCellsPerHit;
  using OuterHitOfCell = caConstants::OuterHitOfCell;
  using CellNeighbors = caConstants::CellNeighbors;
  using CellTracks = caConstants::CellTracks;
  using CellNeighborsVector = caConstants::CellNeighborsVector;
  using CellTracksVector = caConstants::CellTracksVector;

  using Hits = TrackingRecHit2DSOAView;
  using hindex_type = Hits::hindex_type;

  using TmpTuple = cms::cuda::VecArray<uint32_t, 6>;

  using HitContainer = pixelTrack::HitContainer;
  using Quality = pixelTrack::Quality;
  static constexpr auto bad = pixelTrack::Quality::bad;

  GPUCACell() = default;

  __device__ __forceinline__ void init(CellNeighborsVector& cellNeighbors,
                                       CellTracksVector& cellTracks,
                                       Hits const& hh,
                                       int layerPairId,
                                       int doubletId,
                                       hindex_type innerHitId,
                                       hindex_type outerHitId) {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theDoubletId_ = doubletId;
    theLayerPairId_ = layerPairId;
    theUsed_ = 0;

    // optimization that depends on access pattern
    theInnerZ = hh.zGlobal(innerHitId);
    theInnerR = hh.rGlobal(innerHitId);

    // link to default empty
    theOuterNeighbors = &cellNeighbors[0];
    theTracks = &cellTracks[0];
    assert(outerNeighbors().empty());
    assert(tracks().empty());
  }

  __device__ __forceinline__ int addOuterNeighbor(CellNeighbors::value_t t, CellNeighborsVector& cellNeighbors) {
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

  __device__ __forceinline__ int addTrack(CellTracks::value_t t, CellTracksVector& cellTracks) {
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
  __device__ __forceinline__ float inner_x(Hits const& hh) const { return hh.xGlobal(theInnerHitId); }
  __device__ __forceinline__ float outer_x(Hits const& hh) const { return hh.xGlobal(theOuterHitId); }
  __device__ __forceinline__ float inner_y(Hits const& hh) const { return hh.yGlobal(theInnerHitId); }
  __device__ __forceinline__ float outer_y(Hits const& hh) const { return hh.yGlobal(theOuterHitId); }
  __device__ __forceinline__ float inner_z(Hits const& hh) const { return theInnerZ; }
  // { return hh.zGlobal(theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float outer_z(Hits const& hh) const { return hh.zGlobal(theOuterHitId); }
  __device__ __forceinline__ float inner_r(Hits const& hh) const { return theInnerR; }
  // { return hh.rGlobal(theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float outer_r(Hits const& hh) const { return hh.rGlobal(theOuterHitId); }

  __device__ __forceinline__ auto inner_iphi(Hits const& hh) const { return hh.iphi(theInnerHitId); }
  __device__ __forceinline__ auto outer_iphi(Hits const& hh) const { return hh.iphi(theOuterHitId); }

  __device__ __forceinline__ float inner_detIndex(Hits const& hh) const { return hh.detectorIndex(theInnerHitId); }
  __device__ __forceinline__ float outer_detIndex(Hits const& hh) const { return hh.detectorIndex(theOuterHitId); }

  constexpr unsigned int inner_hit_id() const { return theInnerHitId; }
  constexpr unsigned int outer_hit_id() const { return theOuterHitId; }

  __device__ void print_cell() const {
    printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
           theDoubletId_,
           theLayerPairId_,
           theInnerHitId,
           theOuterHitId);
  }

  __device__ bool check_alignment(Hits const& hh,
                                  GPUCACell const& otherCell,
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
    auto isBarrel = otherCell.outer_detIndex(hh) < caConstants::last_barrel_detIndex;
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
                              otherCell.inner_detIndex(hh) < caConstants::last_bpix1_detIndex ? dcaCutInnerTriplet
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

  __device__ inline bool dcaCut(Hits const& hh,
                                GPUCACell const& otherCell,
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

  __device__ inline bool hole0(Hits const& hh, GPUCACell const& innerCell) const {
    using caConstants::first_ladder_bpx0;
    using caConstants::max_ladder_bpx0;
    using caConstants::module_length_bpx0;
    using caConstants::module_tolerance_bpx0;
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

  __device__ inline bool hole4(Hits const& hh, GPUCACell const& innerCell) const {
    using caConstants::first_ladder_bpx4;
    using caConstants::max_ladder_bpx4;
    using caConstants::module_length_bpx4;
    using caConstants::module_tolerance_bpx4;
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
  __device__ inline void find_ntuplets(Hits const& hh,
                                       GPUCACell* __restrict__ cells,
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

    tmpNtuplet.push_back_unsafe(theDoubletId_);
    assert(tmpNtuplet.size() <= 4);

    bool last = true;
    for (unsigned int otherCell : outerNeighbors()) {
      if (cells[otherCell].theDoubletId_ < 0)
        continue;  // killed by earlyFishbone
      last = false;
      cells[otherCell].find_ntuplets(
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
          hindex_type hits[6];
          auto nh = 0U;
          for (auto c : tmpNtuplet) {
            hits[nh++] = cells[c].theInnerHitId;
          }
          hits[nh] = theOuterHitId;
          auto it = foundNtuplets.bulkFill(apc, hits, tmpNtuplet.size() + 1);
          if (it >= 0) {  // if negative is overflow....
            for (auto c : tmpNtuplet)
              cells[c].addTrack(it, cellTracks);
            quality[it] = bad;  // initialize to bad
          }
        }
      }
    }
    tmpNtuplet.pop_back();
    assert(tmpNtuplet.size() < 4);
  }

  // Cell status management
  __device__ __forceinline__ void kill() { theDoubletId_ = -1; }
  __device__ __forceinline__ bool isKilled() const { return theDoubletId_ < 0; }

  __device__ __forceinline__ int16_t layerPairId() const { return theLayerPairId_; }

  __device__ __forceinline__ bool unused() const { return !theUsed_; }
  __device__ __forceinline__ void setUsedBit(uint16_t bit) { theUsed_ |= bit; }

private:
  CellNeighbors* theOuterNeighbors;
  CellTracks* theTracks;

  int32_t theDoubletId_;
  int16_t theLayerPairId_;
  uint16_t theUsed_;  // tbd

  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
