#pragma once

#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include <cstdint>
#include <cmath>

#include<cassert>

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

    float x0,y0,z0;  // the vertex in the local coord of the detector

    Frame frame;

  };


  struct ParamsOnGPU {
    CommonParams * m_commonParams;
    DetParams * m_detParams;

    constexpr
    CommonParams const & commonParams() const {return *m_commonParams;}
    constexpr
    DetParams const &  detParams(int i) const {return m_detParams[i];}

  };

   // SOA!  (on device)
  template<uint32_t N>
  struct ClusParamsT {
    uint32_t minRow[N];
    uint32_t maxRow[N];
    uint32_t minCol[N];
    uint32_t maxCol[N];

    int32_t Q_f_X[N];
    int32_t Q_l_X[N];
    int32_t Q_f_Y[N];
    int32_t Q_l_Y[N];
    
    int32_t charge[N];

    float xpos[N];
    float ypos[N];
  };


  constexpr uint32_t MaxClusInModule=256;
  using ClusParams = ClusParamsT<256>;

  constexpr inline
  void computeAnglesFromDet(DetParams const & detParams, float const x, float const y, float & cotalpha, float & cotbeta) {
    // x,y local position on det
    auto gvx = x - detParams.x0;
    auto gvy = y - detParams.y0;
    auto gvz = -1.f/detParams.z0;
    //  normalization not required as only ratio used...
    // calculate angles
    cotalpha = gvx*gvz;
    cotbeta  = gvy*gvz;  
  }

  constexpr inline
  float correction( 
                         int sizeM1,
                         int Q_f,              //!< Charge in the first pixel.
                         int Q_l,              //!< Charge in the last pixel.
                         uint16_t upper_edge_first_pix, //!< As the name says.
                         uint16_t lower_edge_last_pix,  //!< As the name says.
                         float lorentz_shift,   //!< L-shift at half thickness
                         float theThickness,   //detector thickness
                         float cot_angle,        //!< cot of alpha_ or beta_
                         float pitch,            //!< thePitchX or thePitchY
                         bool first_is_big,       //!< true if the first is big
                         bool last_is_big        //!< true if the last is big
                   )
{
   if (0==sizeM1) return 0;  // size1
   float W_eff=0;
   bool simple=true;
   if (1==sizeM1) {   // size 2   
     //--- Width of the clusters minus the edge (first and last) pixels.
     //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
     // assert(lower_edge_last_pix>=upper_edge_first_pix);
     auto W_inner      =  pitch * float(lower_edge_last_pix-upper_edge_first_pix);  // in cm

     //--- Predicted charge width from geometry
     auto W_pred = theThickness * cot_angle                     // geometric correction (in cm)
                    - lorentz_shift;                    // (in cm) &&& check fpix!
   
     W_eff = std::abs( W_pred ) - W_inner;

     //--- If the observed charge width is inconsistent with the expectations
     //--- based on the track, do *not* use W_pred-W_innner.  Instead, replace
     //--- it with an *average* effective charge width, which is the average
     //--- length of the edge pixels.
     //
     simple = ( W_eff < 0.0f ) | ( W_eff > pitch ); // this produces "large" regressions for very small	numeric differences...

   }
   if (simple) {
     //--- Total length of the two edge pixels (first+last)
     float sum_of_edge = 2.0f;
     if (first_is_big) sum_of_edge += 1.0f;
     if (last_is_big)  sum_of_edge += 1.0f;
     W_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
   }
   
   
   //--- Finally, compute the position in this projection
   float Qdiff = Q_l - Q_f;
   float Qsum  = Q_l + Q_f;
   
   //--- Temporary fix for clusters with both first and last pixel with charge = 0
   if(Qsum==0) Qsum=1.0f;
   return 0.5f*(Qdiff/Qsum) * W_eff;   

  }

  constexpr inline
  void position(CommonParams const & comParams, DetParams const & detParams, ClusParams & cp, uint32_t ic) {

   //--- Upper Right corner of Lower Left pixel -- in measurement frame
   uint16_t llx = cp.minRow[ic]+1;
   uint16_t lly = cp.minCol[ic]+1;
   
   //--- Lower Left corner of Upper Right pixel -- in measurement frame
   uint16_t urx = cp.maxRow[ic];
   uint16_t ury = cp.maxCol[ic];
   
   auto llxl = phase1PixelTopology::localX(llx);   
   auto llyl = phase1PixelTopology::localY(lly);
   auto urxl = phase1PixelTopology::localX(urx);
   auto uryl = phase1PixelTopology::localY(ury);

   auto mx = llxl+urxl;
   auto my = llyl+uryl;   

   // apply the lorentz offset correction
   auto xPos = detParams.shiftX + comParams.thePitchX*(0.5f*float(mx)+float(phase1PixelTopology::xOffset));
   auto yPos = detParams.shiftY + comParams.thePitchY*(0.5f*float(my)+float(phase1PixelTopology::yOffset));
 
   float cotalpha=0, cotbeta=0;


   computeAnglesFromDet(detParams, xPos,  yPos, cotalpha, cotbeta);

   auto thickness = detParams.isBarrel ? comParams.theThicknessB : comParams.theThicknessE;

   auto xcorr = correction(
                            cp.maxRow[ic]-cp.minRow[ic],
                            cp.Q_f_X[ic], cp.Q_l_X[ic],
                            llxl, urxl,
                            detParams.chargeWidthX,   // lorentz shift in cm
                            thickness,
                            cotalpha,
                            comParams.thePitchX,
                            phase1PixelTopology::isBigPixX( cp.minRow[ic] ),
                            phase1PixelTopology::isBigPixX( cp.maxRow[ic] )
                           );   


   auto ycorr = correction(
                            cp.maxCol[ic]-cp.minCol[ic],
                            cp.Q_f_Y[ic], cp.Q_l_Y[ic],
                            llyl, uryl,
                            detParams.chargeWidthY,   // lorentz shift in cm
                            thickness,
                            cotbeta,
                            comParams.thePitchY,
                            phase1PixelTopology::isBigPixY( cp.minCol[ic] ),
                            phase1PixelTopology::isBigPixY( cp.maxCol[ic] )
                           );

   cp.xpos[ic]=xPos+xcorr;
   cp.ypos[ic]=yPos+ycorr;

  }

}
