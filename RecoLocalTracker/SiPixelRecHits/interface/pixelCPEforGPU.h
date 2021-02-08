#ifndef RecoLocalTracker_SiPixelRecHits_pixelCPEforGPU_h
#define RecoLocalTracker_SiPixelRecHits_pixelCPEforGPU_h

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

namespace pixelCPEforGPU {

  using Frame = SOAFrame<float>;
  using Rotation = SOARotation<float>;

  // all modules are identical!
  struct CommonParams {
    float theThicknessB;
    float theThicknessE;
    float thePitchX;
    float thePitchY;
  };

  struct DetParams {
    bool isBarrel;
    bool isPosZ;
    uint16_t layer;
    uint16_t index;
    uint32_t rawId;

    float shiftX;
    float shiftY;
    float chargeWidthX;
    float chargeWidthY;
    uint16_t pixmx;  // max pix charge

    float x0, y0, z0;  // the vertex in the local coord of the detector

    float sx[3], sy[3];  // the errors...

    Frame frame;
  };

  using phase1PixelTopology::AverageGeometry;

  struct LayerGeometry {
    uint32_t layerStart[phase1PixelTopology::numberOfLayers + 1];
    uint8_t layer[phase1PixelTopology::layerIndexSize];
  };

  struct ParamsOnGPU {
    CommonParams const* m_commonParams;
    DetParams const* m_detParams;
    LayerGeometry const* m_layerGeometry;
    AverageGeometry const* m_averageGeometry;

    constexpr CommonParams const& __restrict__ commonParams() const {
      CommonParams const* __restrict__ l = m_commonParams;
      return *l;
    }
    constexpr DetParams const& __restrict__ detParams(int i) const {
      DetParams const* __restrict__ l = m_detParams;
      return l[i];
    }
    constexpr LayerGeometry const& __restrict__ layerGeometry() const { return *m_layerGeometry; }
    constexpr AverageGeometry const& __restrict__ averageGeometry() const { return *m_averageGeometry; }

    __device__ uint8_t layer(uint16_t id) const {
      return __ldg(m_layerGeometry->layer + id / phase1PixelTopology::maxModuleStride);
    };
  };

  // SOA (on device)
  template <uint32_t N>
  struct ClusParamsT {
    uint32_t minRow[N];
    uint32_t maxRow[N];
    uint32_t minCol[N];
    uint32_t maxCol[N];

    int32_t q_f_X[N];
    int32_t q_l_X[N];
    int32_t q_f_Y[N];
    int32_t q_l_Y[N];

    int32_t charge[N];

    float xpos[N];
    float ypos[N];

    float xerr[N];
    float yerr[N];

    int16_t xsize[N];  // clipped at 127 if negative is edge....
    int16_t ysize[N];
  };

  constexpr int32_t MaxHitsInIter = gpuClustering::maxHitsInIter();
  using ClusParams = ClusParamsT<MaxHitsInIter>;

  constexpr inline void computeAnglesFromDet(
      DetParams const& __restrict__ detParams, float const x, float const y, float& cotalpha, float& cotbeta) {
    // x,y local position on det
    auto gvx = x - detParams.x0;
    auto gvy = y - detParams.y0;
    auto gvz = -1.f / detParams.z0;
    // normalization not required as only ratio used...
    // calculate angles
    cotalpha = gvx * gvz;
    cotbeta = gvy * gvz;
  }

  constexpr inline float correction(int sizeM1,
                                    int q_f,                        //!< Charge in the first pixel.
                                    int q_l,                        //!< Charge in the last pixel.
                                    uint16_t upper_edge_first_pix,  //!< As the name says.
                                    uint16_t lower_edge_last_pix,   //!< As the name says.
                                    float lorentz_shift,            //!< L-shift at half thickness
                                    float theThickness,             //detector thickness
                                    float cot_angle,                //!< cot of alpha_ or beta_
                                    float pitch,                    //!< thePitchX or thePitchY
                                    bool first_is_big,              //!< true if the first is big
                                    bool last_is_big)               //!< true if the last is big
  {
    if (0 == sizeM1)  // size 1
      return 0;

    float w_eff = 0;
    bool simple = true;
    if (1 == sizeM1) {  // size 2
      //--- Width of the clusters minus the edge (first and last) pixels.
      //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
      // assert(lower_edge_last_pix >= upper_edge_first_pix);
      auto w_inner = pitch * float(lower_edge_last_pix - upper_edge_first_pix);  // in cm

      //--- Predicted charge width from geometry
      auto w_pred = theThickness * cot_angle  // geometric correction (in cm)
                    - lorentz_shift;          // (in cm) &&& check fpix!

      w_eff = std::abs(w_pred) - w_inner;

      //--- If the observed charge width is inconsistent with the expectations
      //--- based on the track, do *not* use w_pred-w_inner.  Instead, replace
      //--- it with an *average* effective charge width, which is the average
      //--- length of the edge pixels.

      // this can produce "large" regressions for very small numeric differences
      simple = (w_eff < 0.0f) | (w_eff > pitch);
    }

    if (simple) {
      //--- Total length of the two edge pixels (first+last)
      float sum_of_edge = 2.0f;
      if (first_is_big)
        sum_of_edge += 1.0f;
      if (last_is_big)
        sum_of_edge += 1.0f;
      w_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
    }

    //--- Finally, compute the position in this projection
    float qdiff = q_l - q_f;
    float qsum = q_l + q_f;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (qsum == 0)
      qsum = 1.0f;

    return 0.5f * (qdiff / qsum) * w_eff;
  }

  constexpr inline void position(CommonParams const& __restrict__ comParams,
                                 DetParams const& __restrict__ detParams,
                                 ClusParams& cp,
                                 uint32_t ic) {
    //--- Upper Right corner of Lower Left pixel -- in measurement frame
    uint16_t llx = cp.minRow[ic] + 1;
    uint16_t lly = cp.minCol[ic] + 1;

    //--- Lower Left corner of Upper Right pixel -- in measurement frame
    uint16_t urx = cp.maxRow[ic];
    uint16_t ury = cp.maxCol[ic];

    auto llxl = phase1PixelTopology::localX(llx);
    auto llyl = phase1PixelTopology::localY(lly);
    auto urxl = phase1PixelTopology::localX(urx);
    auto uryl = phase1PixelTopology::localY(ury);

    auto mx = llxl + urxl;
    auto my = llyl + uryl;

    auto xsize = int(urxl) + 2 - int(llxl);
    auto ysize = int(uryl) + 2 - int(llyl);
    assert(xsize >= 0);  // 0 if bixpix...
    assert(ysize >= 0);

    if (phase1PixelTopology::isBigPixX(cp.minRow[ic]))
      ++xsize;
    if (phase1PixelTopology::isBigPixX(cp.maxRow[ic]))
      ++xsize;
    if (phase1PixelTopology::isBigPixY(cp.minCol[ic]))
      ++ysize;
    if (phase1PixelTopology::isBigPixY(cp.maxCol[ic]))
      ++ysize;

    int unbalanceX = 8. * std::abs(float(cp.q_f_X[ic] - cp.q_l_X[ic])) / float(cp.q_f_X[ic] + cp.q_l_X[ic]);
    int unbalanceY = 8. * std::abs(float(cp.q_f_Y[ic] - cp.q_l_Y[ic])) / float(cp.q_f_Y[ic] + cp.q_l_Y[ic]);
    xsize = 8 * xsize - unbalanceX;
    ysize = 8 * ysize - unbalanceY;

    cp.xsize[ic] = std::min(xsize, 1023);
    cp.ysize[ic] = std::min(ysize, 1023);

    if (cp.minRow[ic] == 0 || cp.maxRow[ic] == phase1PixelTopology::lastRowInModule)
      cp.xsize[ic] = -cp.xsize[ic];
    if (cp.minCol[ic] == 0 || cp.maxCol[ic] == phase1PixelTopology::lastColInModule)
      cp.ysize[ic] = -cp.ysize[ic];

    // apply the lorentz offset correction
    auto xPos = detParams.shiftX + comParams.thePitchX * (0.5f * float(mx) + float(phase1PixelTopology::xOffset));
    auto yPos = detParams.shiftY + comParams.thePitchY * (0.5f * float(my) + float(phase1PixelTopology::yOffset));

    float cotalpha = 0, cotbeta = 0;

    computeAnglesFromDet(detParams, xPos, yPos, cotalpha, cotbeta);

    auto thickness = detParams.isBarrel ? comParams.theThicknessB : comParams.theThicknessE;

    auto xcorr = correction(cp.maxRow[ic] - cp.minRow[ic],
                            cp.q_f_X[ic],
                            cp.q_l_X[ic],
                            llxl,
                            urxl,
                            detParams.chargeWidthX,  // lorentz shift in cm
                            thickness,
                            cotalpha,
                            comParams.thePitchX,
                            phase1PixelTopology::isBigPixX(cp.minRow[ic]),
                            phase1PixelTopology::isBigPixX(cp.maxRow[ic]));

    auto ycorr = correction(cp.maxCol[ic] - cp.minCol[ic],
                            cp.q_f_Y[ic],
                            cp.q_l_Y[ic],
                            llyl,
                            uryl,
                            detParams.chargeWidthY,  // lorentz shift in cm
                            thickness,
                            cotbeta,
                            comParams.thePitchY,
                            phase1PixelTopology::isBigPixY(cp.minCol[ic]),
                            phase1PixelTopology::isBigPixY(cp.maxCol[ic]));

    cp.xpos[ic] = xPos + xcorr;
    cp.ypos[ic] = yPos + ycorr;
  }

  constexpr inline void errorFromSize(CommonParams const& __restrict__ comParams,
                                      DetParams const& __restrict__ detParams,
                                      ClusParams& cp,
                                      uint32_t ic) {
    // Edge cluster errors
    cp.xerr[ic] = 0.0050;
    cp.yerr[ic] = 0.0085;

    // FIXME these are errors form Run1
    constexpr float xerr_barrel_l1[] = {0.00115, 0.00120, 0.00088};
    constexpr float xerr_barrel_l1_def = 0.00200;  // 0.01030;
    constexpr float yerr_barrel_l1[] = {
        0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240};
    constexpr float yerr_barrel_l1_def = 0.00210;
    constexpr float xerr_barrel_ln[] = {0.00115, 0.00120, 0.00088};
    constexpr float xerr_barrel_ln_def = 0.00200;  // 0.01030;
    constexpr float yerr_barrel_ln[] = {
        0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240};
    constexpr float yerr_barrel_ln_def = 0.00210;
    constexpr float xerr_endcap[] = {0.0020, 0.0020};
    constexpr float xerr_endcap_def = 0.0020;
    constexpr float yerr_endcap[] = {0.00210};
    constexpr float yerr_endcap_def = 0.00210;

    auto sx = cp.maxRow[ic] - cp.minRow[ic];
    auto sy = cp.maxCol[ic] - cp.minCol[ic];

    // is edgy ?
    bool isEdgeX = cp.minRow[ic] == 0 or cp.maxRow[ic] == phase1PixelTopology::lastRowInModule;
    bool isEdgeY = cp.minCol[ic] == 0 or cp.maxCol[ic] == phase1PixelTopology::lastColInModule;
    // is one and big?
    bool isBig1X = (0 == sx) && phase1PixelTopology::isBigPixX(cp.minRow[ic]);
    bool isBig1Y = (0 == sy) && phase1PixelTopology::isBigPixY(cp.minCol[ic]);

    if (!isEdgeX && !isBig1X) {
      if (not detParams.isBarrel) {
        cp.xerr[ic] = sx < std::size(xerr_endcap) ? xerr_endcap[sx] : xerr_endcap_def;
      } else if (detParams.layer == 1) {
        cp.xerr[ic] = sx < std::size(xerr_barrel_l1) ? xerr_barrel_l1[sx] : xerr_barrel_l1_def;
      } else {
        cp.xerr[ic] = sx < std::size(xerr_barrel_ln) ? xerr_barrel_ln[sx] : xerr_barrel_ln_def;
      }
    }

    if (!isEdgeY && !isBig1Y) {
      if (not detParams.isBarrel) {
        cp.yerr[ic] = sy < std::size(yerr_endcap) ? yerr_endcap[sy] : yerr_endcap_def;
      } else if (detParams.layer == 1) {
        cp.yerr[ic] = sy < std::size(yerr_barrel_l1) ? yerr_barrel_l1[sy] : yerr_barrel_l1_def;
      } else {
        cp.yerr[ic] = sy < std::size(yerr_barrel_ln) ? yerr_barrel_ln[sy] : yerr_barrel_ln_def;
      }
    }
  }

  constexpr inline void errorFromDB(CommonParams const& __restrict__ comParams,
                                    DetParams const& __restrict__ detParams,
                                    ClusParams& cp,
                                    uint32_t ic) {
    // Edge cluster errors
    cp.xerr[ic] = 0.0050f;
    cp.yerr[ic] = 0.0085f;

    auto sx = cp.maxRow[ic] - cp.minRow[ic];
    auto sy = cp.maxCol[ic] - cp.minCol[ic];

    // is edgy ?
    bool isEdgeX = cp.minRow[ic] == 0 or cp.maxRow[ic] == phase1PixelTopology::lastRowInModule;
    bool isEdgeY = cp.minCol[ic] == 0 or cp.maxCol[ic] == phase1PixelTopology::lastColInModule;
    // is one and big?
    uint32_t ix = (0 == sx);
    uint32_t iy = (0 == sy);
    ix += (0 == sx) && phase1PixelTopology::isBigPixX(cp.minRow[ic]);
    iy += (0 == sy) && phase1PixelTopology::isBigPixY(cp.minCol[ic]);

    if (not isEdgeX)
      cp.xerr[ic] = detParams.sx[ix];
    if (not isEdgeY)
      cp.yerr[ic] = detParams.sy[iy];
  }

}  // namespace pixelCPEforGPU

#endif  // RecoLocalTracker_SiPixelRecHits_pixelCPEforGPU_h
