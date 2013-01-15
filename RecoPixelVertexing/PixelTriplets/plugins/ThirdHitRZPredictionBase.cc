#include "ThirdHitRZPredictionBase.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

ThirdHitRZPredictionBase::ThirdHitRZPredictionBase() : 
  theBarrel(false), theForward(false), theTolerance(0.,0.)
{}

ThirdHitRZPredictionBase::ThirdHitRZPredictionBase(
  float tolerance, const DetLayer* layer)
  : theBarrel(false), theForward(false), theTolerance(tolerance, tolerance)
{
    if (layer) initLayer(layer);
}

void ThirdHitRZPredictionBase::initLayer(const DetLayer *layer)
{
  if (layer->location() == GeomDetEnumerators::barrel) {
    theBarrel = true;
    theForward = false;
    const BarrelDetLayer& bl = reinterpret_cast<const BarrelDetLayer&>(*layer);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    theDetRange = Range(radius-halfThickness, radius+halfThickness);
    float maxZ = bl.surface().bounds().length()/2;
    theDetSize = Range(-maxZ, maxZ);
  } else if (layer->location() == GeomDetEnumerators::endcap) {
    theBarrel= false;
    theForward = true;
    const ForwardDetLayer& fl = reinterpret_cast<const ForwardDetLayer&>(*layer);
    float halfThickness  = fl.surface().bounds().thickness()/2;
    float zLayer = fl.position().z() ;
    theDetRange = Range(zLayer-halfThickness, zLayer+halfThickness);
    const SimpleDiskBounds& diskRadialBounds =
                 static_cast<const SimpleDiskBounds &>(fl.surface().bounds());
    theDetSize = Range(diskRadialBounds.innerRadius(), diskRadialBounds.outerRadius());
  } else {
    theBarrel = theForward = false;

  }
}
