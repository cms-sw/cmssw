#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

template <class T> T sqr( T t) {return t*t;}

ThirdHitRZPrediction::ThirdHitRZPrediction() : 
  theBarrel(false), theForward(false), theTolerance(0.,0.)
{}

ThirdHitRZPrediction::ThirdHitRZPrediction(
  const PixelRecoLineRZ &line, float tolerance, const DetLayer* layer)
  : theBarrel(false), theForward(false), theTolerance(tolerance, tolerance), theLine(line)
{
    if (layer) initLayer(layer);
}

ThirdHitRZPrediction::Range ThirdHitRZPrediction::operator()(const DetLayer *layer)
{
  if (layer) initLayer(layer);

  float v1, v2;
  if (theBarrel) {
    v1 = theLine.zAtR(theDetRange.min());
    v2 = theLine.zAtR(theDetRange.max());
  } else if (theForward) {
    v1 = theLine.rAtZ(theDetRange.min());
    v2 = theLine.rAtZ(theDetRange.max());
  } else
    return Range(0., 0.);

  if (v1 > v2) std::swap(v1,v2);
  float rl = v1-theTolerance.left();
  float rr = v2+theTolerance.right();
  return Range(rl,rr);
}

ThirdHitRZPrediction::Range ThirdHitRZPrediction::operator()(float rORz, const PixelRecoLineRZ &line) const
{
  float v = theBarrel ? line.zAtR(rORz) : line.rAtZ(rORz);
  return Range(v - theTolerance.left(), v + theTolerance.right());
}

void ThirdHitRZPrediction::initLayer(const DetLayer *layer)
{
  if (layer->location() == GeomDetEnumerators::barrel) {
    theBarrel = true;
    theForward = false;
    const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*layer);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    theDetRange = Range(radius-halfThickness, radius+halfThickness);
    float maxZ = bl.surface().bounds().length()/2;
    theDetSize = Range(-maxZ, maxZ);
  } else if (layer->location() == GeomDetEnumerators::endcap) {
    theBarrel= false;
    theForward = true;
    const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*layer);
    float halfThickness  = fl.surface().bounds().thickness()/2;
    float zLayer = fl.position().z() ;
    theDetRange = Range(zLayer-halfThickness, zLayer+halfThickness);
    const SimpleDiskBounds& diskRadialBounds =
                 static_cast<const SimpleDiskBounds &>(fl.surface().bounds());
    theDetSize = Range(diskRadialBounds.innerRadius(), diskRadialBounds.outerRadius());
  }
}
