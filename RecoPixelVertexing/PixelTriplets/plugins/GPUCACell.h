//
// Author: Felice Pantaleo, CERN
//
#ifndef RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
#define RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

struct Quadruplet {
   using hindex_type = siPixelRecHitsHeterogeneousProduct::hindex_type;
   hindex_type hitId[4];
};


class GPUCACell {
public:

  using Hits = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
  using hindex_type = siPixelRecHitsHeterogeneousProduct::hindex_type;

  GPUCACell() = default;
#ifdef __CUDACC__

  __device__ __forceinline__
  void init(Hits const & hh,
      int layerPairId, int doubletId,  
      hindex_type innerHitId, hindex_type outerHitId)
  {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theDoubletId = doubletId;
    theLayerPairId = layerPairId;

    theInnerZ = __ldg(hh.zg_d+innerHitId);
    theInnerR = __ldg(hh.rg_d+innerHitId);
    theOuterNeighbors.reset();
  }

  __device__ __forceinline__ float get_inner_x(Hits const & hh) const { return __ldg(hh.xg_d+theInnerHitId); }
  __device__ __forceinline__ float get_outer_x(Hits const & hh) const { return __ldg(hh.xg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_y(Hits const & hh) const { return __ldg(hh.yg_d+theInnerHitId); }
  __device__ __forceinline__ float get_outer_y(Hits const & hh) const { return __ldg(hh.yg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_z(Hits const & hh) const { return theInnerZ; } // { return __ldg(hh.zg_d+theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float get_outer_z(Hits const & hh) const { return __ldg(hh.zg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_r(Hits const & hh) const { return theInnerR; } // { return __ldg(hh.rg_d+theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float get_outer_r(Hits const & hh) const { return __ldg(hh.rg_d+theOuterHitId); }

  constexpr unsigned int get_inner_hit_id() const {
    return theInnerHitId;
  }
  constexpr unsigned int get_outer_hit_id() const {
    return theOuterHitId;
  }


  __device__
  void print_cell() const {
    printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: "
           "%d, innerradius %f, outerRadius %f \n",
           theDoubletId, theLayerPairId, theInnerHitId, theOuterHitId
    );
  }


  __device__
  bool check_alignment(Hits const & hh,
      GPUCACell const & otherCell, const float ptmin,
      const float region_origin_x, const float region_origin_y,
      const float region_origin_radius, const float thetaCut,
      const float phiCut, const float hardPtCut) const
  {
    auto ri = get_inner_r(hh);
    auto zi = get_inner_z(hh);

    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);

    auto r1 = otherCell.get_inner_r(hh);
    auto z1 = otherCell.get_inner_z(hh);
    bool aligned = areAlignedRZ(r1, z1, ri, zi, ro, zo, ptmin, thetaCut);
    return (aligned &&
            haveSimilarCurvature(hh, otherCell, ptmin, region_origin_x,
                                 region_origin_y, region_origin_radius, phiCut,
                                 hardPtCut));
  }

  __device__ __forceinline__
  static bool areAlignedRZ(float r1, float z1, float ri, float zi, float ro, float zo,
                                        const float ptmin,
                                        const float thetaCut) {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared =
        radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    float pMin =
        ptmin * std::sqrt(distance_13_squared); // this needs to be divided by
                                                // radius_diff later

    float tan_12_13_half_mul_distance_13_squared =
        fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
    return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
  }

  __device__ 
  bool
  haveSimilarCurvature(Hits const & hh, GPUCACell const & otherCell,
                       const float ptmin, const float region_origin_x,
                       const float region_origin_y,
                       const float region_origin_radius, const float phiCut,
                       const float hardPtCut) const {

    auto x1 = otherCell.get_inner_x(hh);
    auto y1 = otherCell.get_inner_y(hh);

    auto x2 = get_inner_x(hh);
    auto y2 = get_inner_y(hh);

    auto x3 = get_outer_x(hh);
    auto y3 = get_outer_y(hh);

    float distance_13_squared = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
    float tan_12_13_half_mul_distance_13_squared =
        fabs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2));
    // high pt : just straight
    if (tan_12_13_half_mul_distance_13_squared * ptmin <=
        1.0e-4f * distance_13_squared) {

      float distance_3_beamspot_squared =
          (x3 - region_origin_x) * (x3 - region_origin_x) +
          (y3 - region_origin_y) * (y3 - region_origin_y);

      float dot_bs3_13 = ((x1 - x3) * (region_origin_x - x3) +
                          (y1 - y3) * (region_origin_y - y3));
      float proj_bs3_on_13_squared =
          dot_bs3_13 * dot_bs3_13 / distance_13_squared;

      float distance_13_beamspot_squared =
          distance_3_beamspot_squared - proj_bs3_on_13_squared;

      return distance_13_beamspot_squared <
             (region_origin_radius + phiCut) * (region_origin_radius + phiCut);
    } 

    // 87 cm/GeV = 1/(3.8T * 0.3)

    // take less than radius given by the hardPtCut and reject everything below
    float minRadius = hardPtCut * 87.f; // FIXME move out and use real MagField

    auto det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);

    auto offset = x2 * x2 + y2 * y2;

    auto bc = (x1 * x1 + y1 * y1 - offset) * 0.5f;

    auto cd = (offset - x3 * x3 - y3 * y3) * 0.5f;

    auto idet = 1.f / det;

    auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
    auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

    auto radius = std::sqrt((x2 - x_center) * (x2 - x_center) +
                            (y2 - y_center) * (y2 - y_center));

    if (radius < minRadius)
      return false; // hard cut on pt

    auto centers_distance_squared =
        (x_center - region_origin_x) * (x_center - region_origin_x) +
        (y_center - region_origin_y) * (y_center - region_origin_y);
    auto region_origin_radius_plus_tolerance = region_origin_radius + phiCut;
    auto minimumOfIntersectionRange =
        (radius - region_origin_radius_plus_tolerance) *
        (radius - region_origin_radius_plus_tolerance);

    if (centers_distance_squared >= minimumOfIntersectionRange) {
      auto maximumOfIntersectionRange =
          (radius + region_origin_radius_plus_tolerance) *
          (radius + region_origin_radius_plus_tolerance);
      return centers_distance_squared <= maximumOfIntersectionRange;
    }

    return false;
  }

  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.

// #ifdef __CUDACC__

  __device__
  inline void find_ntuplets(
      GPUCACell const * __restrict__ cells,
      GPU::SimpleVector<Quadruplet> *foundNtuplets,
      GPU::VecArray<hindex_type,3> &tmpNtuplet,
      const unsigned int minHitsPerNtuplet) const
  {
    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold

    tmpNtuplet.push_back_unsafe(theInnerHitId);
    assert(tmpNtuplet.size()<=3);

    if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet-1) {
      Quadruplet tmpQuadruplet;
      for (unsigned int i = 0; i < minHitsPerNtuplet-1; ++i) {
        tmpQuadruplet.hitId[i] = tmpNtuplet[i];
      }
      tmpQuadruplet.hitId[minHitsPerNtuplet-1] = theOuterHitId;
      foundNtuplets->push_back(tmpQuadruplet);
    }
    else {
      for (int j = 0; j < theOuterNeighbors.size(); ++j) {
        auto otherCell = theOuterNeighbors[j];
        cells[otherCell].find_ntuplets(cells, foundNtuplets, tmpNtuplet,
                                       minHitsPerNtuplet);
      }
    }
    tmpNtuplet.pop_back();
    assert(tmpNtuplet.size() < 3);
  }

#endif // __CUDACC__

  GPU::VecArray< unsigned int, 40> theOuterNeighbors;

  int theDoubletId;
  int theLayerPairId;

private:
  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
};

#endif // RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
