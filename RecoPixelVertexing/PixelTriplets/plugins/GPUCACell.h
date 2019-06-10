#ifndef RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
#define RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h

//
// Author: Felice Pantaleo, CERN
//

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CircleEq.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

class GPUCACell {
public:
  using ptrAsInt = unsigned long long;

  static constexpr int maxCellsPerHit = CAConstants::maxCellsPerHit();
  using OuterHitOfCell = CAConstants::OuterHitOfCell;
  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  using Hits = TrackingRecHit2DSOAView;
  using hindex_type = Hits::hindex_type;

  using TmpTuple = GPU::VecArray<uint32_t, 6>;

  using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  GPUCACell() = default;
#ifdef __CUDACC__

  __device__ __forceinline__ void init(CellNeighborsVector& cellNeighbors,
                                       CellTracksVector& cellTracks,
                                       Hits const& hh,
                                       int layerPairId,
                                       int doubletId,
                                       hindex_type innerHitId,
                                       hindex_type outerHitId) {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theDoubletId = doubletId;
    theLayerPairId = layerPairId;

    theInnerZ = hh.zGlobal(innerHitId);
    theInnerR = hh.rGlobal(innerHitId);

    outerNeighbors().reset();
    tracks().reset();
    assert(outerNeighbors().empty());
    assert(tracks().empty());
  }

  __device__ __forceinline__ int addOuterNeighbor(CellNeighbors::value_t t, CellNeighborsVector& cellNeighbors) {
    return outerNeighbors().push_back(t);
  }

  __device__ __forceinline__ int addTrack(CellTracks::value_t t, CellTracksVector& cellTracks) {
    return tracks().push_back(t);
  }

  __device__ __forceinline__ CellTracks& tracks() { return theTracks; }
  __device__ __forceinline__ CellTracks const& tracks() const { return theTracks; }
  __device__ __forceinline__ CellNeighbors& outerNeighbors() { return theOuterNeighbors; }
  __device__ __forceinline__ CellNeighbors const& outerNeighbors() const { return theOuterNeighbors; }
  __device__ __forceinline__ float get_inner_x(Hits const& hh) const { return hh.xGlobal(theInnerHitId); }
  __device__ __forceinline__ float get_outer_x(Hits const& hh) const { return hh.xGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_y(Hits const& hh) const { return hh.yGlobal(theInnerHitId); }
  __device__ __forceinline__ float get_outer_y(Hits const& hh) const { return hh.yGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_z(Hits const& hh) const {
    return theInnerZ;
  }  // { return hh.zGlobal(theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float get_outer_z(Hits const& hh) const { return hh.zGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_r(Hits const& hh) const {
    return theInnerR;
  }  // { return hh.rGlobal(theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float get_outer_r(Hits const& hh) const { return hh.rGlobal(theOuterHitId); }

  __device__ __forceinline__ auto get_inner_iphi(Hits const& hh) const { return hh.iphi(theInnerHitId); }
  __device__ __forceinline__ auto get_outer_iphi(Hits const& hh) const { return hh.iphi(theOuterHitId); }

  __device__ __forceinline__ float get_inner_detIndex(Hits const& hh) const { return hh.detectorIndex(theInnerHitId); }
  __device__ __forceinline__ float get_outer_detIndex(Hits const& hh) const { return hh.detectorIndex(theOuterHitId); }

  constexpr unsigned int get_inner_hit_id() const { return theInnerHitId; }
  constexpr unsigned int get_outer_hit_id() const { return theOuterHitId; }

  __device__ void print_cell() const {
    printf(
        "printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: "
        "%d, innerradius %f, outerRadius %f \n",
        theDoubletId,
        theLayerPairId,
        theInnerHitId,
        theOuterHitId);
  }

  __device__ bool check_alignment(Hits const& hh,
                                  GPUCACell const& otherCell,
                                  const float ptmin,
                                  const float hardCurvCut,
                                  const float CAThetaCutBarrel,
                                  const float CAThetaCutForward,
                                  const float dcaCutInnerTriplet,
                                  const float dcaCutOuterTriplet) const {
    // detIndex of the layerStart for the Phase1 Pixel Detector:
    // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
    // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
    constexpr uint32_t last_bpix1_detIndex = 96;
    constexpr uint32_t last_barrel_detIndex = 1184;
    auto ri = get_inner_r(hh);
    auto zi = get_inner_z(hh);

    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);

    auto r1 = otherCell.get_inner_r(hh);
    auto z1 = otherCell.get_inner_z(hh);
    auto isBarrel = otherCell.get_outer_detIndex(hh) < last_barrel_detIndex;
    bool aligned = areAlignedRZ(r1,
                                z1,
                                ri,
                                zi,
                                ro,
                                zo,
                                ptmin,
                                isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
    return (aligned &&
            dcaCut(hh,
                   otherCell,
                   otherCell.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet : dcaCutOuterTriplet,
                   hardCurvCut));  // FIXME tune cuts
                                   // region_origin_radius_plus_tolerance,  hardCurvCut));
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
    auto x1 = otherCell.get_inner_x(hh);
    auto y1 = otherCell.get_inner_y(hh);

    auto x2 = get_inner_x(hh);
    auto y2 = get_inner_y(hh);

    auto x3 = get_outer_x(hh);
    auto y3 = get_outer_y(hh);

    CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

    if (eq.curvature() > maxCurv)
      return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
  }

  __device__ inline bool hole(Hits const& hh, GPUCACell const& innerCell) const {
    constexpr uint32_t max_ladder_bpx4 = 64;
    constexpr float radius_even_ladder = 15.815f;
    constexpr float radius_odd_ladder = 16.146f;
    constexpr float ladder_length = 6.7f;
    constexpr float ladder_tolerance = 0.2f;
    constexpr float barrel_z_length = 26.f;
    constexpr float forward_z_begin = 32.f;
    int p = get_outer_iphi(hh);
    if (p < 0)
      p += std::numeric_limits<unsigned short>::max();
    p = (max_ladder_bpx4 * p) / std::numeric_limits<unsigned short>::max();
    p %= 2;
    float r4 = p == 0 ? radius_even_ladder : radius_odd_ladder;  // later on from geom
    auto ri = innerCell.get_inner_r(hh);
    auto zi = innerCell.get_inner_z(hh);
    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);
    auto z4 = std::abs(zi + (r4 - ri) * (zo - zi) / (ro - ri));
    auto z_in_ladder = z4 - ladder_length * int(z4 / ladder_length);
    auto h = z_in_ladder < ladder_tolerance || z_in_ladder > (ladder_length - ladder_tolerance);
    return h || (z4 > barrel_z_length && z4 < forward_z_begin);
  }

  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.

  template <typename CM>
  __device__ inline void find_ntuplets(Hits const& hh,
                                       GPUCACell* __restrict__ cells,
                                       CellTracksVector& cellTracks,
                                       TuplesOnGPU::Container& foundNtuplets,
                                       AtomicPairCounter& apc,
                                       CM& tupleMultiplicity,
                                       TmpTuple& tmpNtuplet,
                                       const unsigned int minHitsPerNtuplet) const {
    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold

    tmpNtuplet.push_back_unsafe(theDoubletId);
    assert(tmpNtuplet.size() <= 4);

    if (outerNeighbors().size() > 0) {
      for (int j = 0; j < outerNeighbors().size(); ++j) {
        auto otherCell = outerNeighbors()[j];
        cells[otherCell].find_ntuplets(
            hh, cells, cellTracks, foundNtuplets, apc, tupleMultiplicity, tmpNtuplet, minHitsPerNtuplet);
      }
    } else {  // if long enough save...
      if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifndef ALL_TRIPLETS
        // triplets accepted only pointing to the hole
        if (tmpNtuplet.size() >= 3 || hole(hh, cells[tmpNtuplet[0]]))
#endif
        {
          hindex_type hits[6];
          auto nh = 0U;
          for (auto c : tmpNtuplet)
            hits[nh++] = cells[c].theInnerHitId;
          hits[nh] = theOuterHitId;
          auto it = foundNtuplets.bulkFill(apc, hits, tmpNtuplet.size() + 1);
          if (it >= 0) {  // if negative is overflow....
            for (auto c : tmpNtuplet)
              cells[c].addTrack(it, cellTracks);
            tupleMultiplicity.countDirect(tmpNtuplet.size() + 1);
          }
        }
      }
    }
    tmpNtuplet.pop_back();
    assert(tmpNtuplet.size() < 4);
  }

#endif  // __CUDACC__

private:
  CellNeighbors theOuterNeighbors;
  CellTracks theTracks;

public:
  int32_t theDoubletId;
  int32_t theLayerPairId;

private:
  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
