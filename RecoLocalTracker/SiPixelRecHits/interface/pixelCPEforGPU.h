#pragma once

#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include <cstdint>

namespace pixelCPEforGPU {

  using Frame = SOAFrame<float>;

  // all modules are identical!
  struct CommonParam {
    float theThickness;
    float thePitchX;
    float thePitchY;
  };

  struct DetParam {

    bool isBarrel;
    bool isPosZ;
    uint16_t layer;
    unit16_t index;
    uint32_t rawId;

    /*
    float widthLAFractionX;    // Width-LA to Offset-LA in X
    float widthLAFractionY;    // same in Y
    float lorentzShiftInCmX;   // a FULL shift, in cm
    float lorentzShiftInCmY;   // a FULL shift, in cm
    */

    float shiftX;
    float shiftY;

    float x0,y0,z0;  // the vertex in the local coord of the detector

    Frame frame;

  };


  /*

   float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
   float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
   float shiftX = 0.5f*theDetParam.lorentzShiftInCmX;
   float shiftY = 0.5f*theDetParam.lorentzShiftInCmY;

  */


  constexpr inline
  void computeAnglesFromDet(DetParam const & detParam, float const x, float const y, float & cotalpha, float & cotbeta) {
    // x,y local position on det
    auto gvx = x - detParam.x0;
    auto gvy = y  -detParam.y0;
    auto gvz = -1.f/detParam.z0;
    //  normalization not required as only ratio used...
    // calculate angles
    cotalpha = gvx*gvz;
    cotbeta  = gvy*gvz;
  
  }

  constexpr inline
  void medianPosition(DetParam const & detParam,) {

   //--- Upper Right corner of Lower Left pixel -- in measurement frame
   uint16_t llx = minPixelRow+1;
   uint16_t lly = minPixelCol+1;
   
   //--- Lower Left corner of Upper Right pixel -- in measurement frame
   uint16_t urx = maxPixelRow;
   uint16_t ury = maxPixelCol;
   
   auto llxl = phase1PixelTopology::localX(llx);   
   auto llyl = phase1PixelTopology::localY(lly);
   auto urxl = phase1PixelTopology::localX(urx);
   auto uryl = phase1PixelTopology::localY(ury);

   auto mx = llxl+urxl;
   auto my = llyl+uryl;   

   // apply the lorentz offset correction
   xPos = shiftX + theComParam.thePitchX*(0.5f*float(mx)+float(phase1PixelTopology::xOffset));
   yPos = shiftY + thecomParam.thePitchY*(0.5f*float(my)+float(phase1PixelTopology::yOffset));
 
  } 


}
