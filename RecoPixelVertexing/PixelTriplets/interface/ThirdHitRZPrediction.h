#ifndef ThirdHitRZPrediction_H
#define ThirdHitRZPrediction_H

/** predicts a range in r-z for the position of third hit.
 *  the predicted reange is defined by stright line extrapolation/interpolation
 *  from hit pair and the margin defined by hit errors and multiple scattering
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"

class DetLayer;
class OrderedHitPair;
class MultipleScatteringParametrisation;
namespace pixelrecoutilities { class  LongitudinalBendingCorrection; }

class ThirdHitRZPrediction {
public:
  typedef PixelRecoRange<float> Range;
  typedef TkTrackingRegionsMargin<float> Margin;

  ThirdHitRZPrediction(const GlobalPoint &gp1, const GlobalPoint &gp2, 
      float tolerance, const DetLayer* layer=0);

  Range operator()( const DetLayer *layer=0 ); 
  Range operator()(float rORz);

  Range detRange() const { return theDetRange; }

  void initLayer(const DetLayer *layer);
private:


  bool theBarrel, theForward;
  Range theDetRange;
  Margin theTolerance;
  PixelRecoLineRZ theLine;
  
};
#endif
