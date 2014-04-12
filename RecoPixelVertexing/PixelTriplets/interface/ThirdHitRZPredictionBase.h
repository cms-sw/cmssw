#ifndef ThirdHitRZPredictionBase_H
#define ThirdHitRZPredictionBase_H

/** predicts a range in r-z for the position of third hit.
 *  The is the base class with common code, the actual implementation
 *  is interfaced in the ThirdHitRZPrediction template.
 *  The margin is defined by hit errors and multiple scattering.
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"

class DetLayer;
class OrderedHitPair;
class MultipleScatteringParametrisation;

class ThirdHitRZPredictionBase {
public:
  typedef PixelRecoRange<float> Range;
  typedef TkTrackingRegionsMargin<float> Margin;

  ThirdHitRZPredictionBase();
  ThirdHitRZPredictionBase(float tolerance, const DetLayer* layer = 0);

  const Range & detRange() const { return theDetRange; }
  const Range & detSize() const { return theDetSize; }

  void initTolerance(float tolerance) { theTolerance = Margin(tolerance, tolerance); }
  void initLayer(const DetLayer *layer);

protected:
  bool theBarrel, theForward;
  Range theDetRange, theDetSize;
  Margin theTolerance;
};
#endif
