#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"

template <class T> T sqr( T t) {return t*t;}

ThirdHitRZPrediction::ThirdHitRZPrediction(
  const GlobalPoint &gp1, const GlobalPoint &gp2, float tolerance, const DetLayer* layer)
  : theBarrel(false), theForward(false), theTolerance( Margin(tolerance,tolerance) )
{
    PixelRecoPointRZ p1(gp1.perp(), gp1.z());
    PixelRecoPointRZ p2(gp2.perp(), gp2.z());
    float dr = p2.r() - p1.r();
    if ( dr != 0.) {
      theLine = PixelRecoLineRZ(p2, p1);
    }
    else { theLine = PixelRecoLineRZ(p1, 99999.); }
   
    if (layer) initLayer(layer);

}

ThirdHitRZPrediction::Range ThirdHitRZPrediction:: operator()(const DetLayer *layer) 
{
  if (layer) initLayer(layer);

  float v1,v2;
  if (theBarrel) {
    v1 = theLine.zAtR(theDetRange.min());
    v2 = theLine.zAtR(theDetRange.max());
  } else if (theForward) {
    v1 = theLine.rAtZ(theDetRange.min());
    v2 = theLine.rAtZ(theDetRange.max());
  } else return Range(0.,0.); 

  if (v1 > v2) std::swap(v1,v2);
  float rl = v1-theTolerance.left();
  float rr = v2+theTolerance.right();
  return Range(rl,rr);
}

ThirdHitRZPrediction::Range ThirdHitRZPrediction::operator()(float rORz)
{
  float v;
  if (theBarrel) {
    v=theLine.zAtR(rORz); 
  } else {
    v=theLine.rAtZ(rORz); 
  }

  float rl = v-theTolerance.left();
  float rr = v+theTolerance.right();
  return Range(rl,rr);
}

void ThirdHitRZPrediction::initLayer(const DetLayer *layer)
{
  if ( layer->location() == GeomDetEnumerators::barrel) {
    theBarrel = true;
    theForward = false;
    const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*layer);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    theDetRange = Range(radius-halfThickness, radius+halfThickness);
  } else if ( layer->location() == GeomDetEnumerators::endcap) {
    theBarrel= false;
    theForward = true;
    const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*layer);
    float halfThickness  = fl.surface().bounds().thickness()/2;
    float zLayer = fl.position().z() ;
    theDetRange = Range(zLayer-halfThickness, zLayer+halfThickness);
  }
}
