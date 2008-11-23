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

class ThirdHitRZPrediction {
public:
  typedef PixelRecoRange<float> Range;
  typedef TkTrackingRegionsMargin<float> Margin;

  ThirdHitRZPrediction();
  ThirdHitRZPrediction(const PixelRecoLineRZ &line, 
      float tolerance, const DetLayer* layer = 0);

  Range operator()(const DetLayer *layer = 0); 
  inline Range operator()(float rORz) const { return (*this)(rORz, theLine); }
  Range operator()(float rORz, const PixelRecoLineRZ &line) const;

  const Range & detRange() const { return theDetRange; }
  const Range & detSize() const { return theDetSize; }

  void initTolerance(float tolerance) {  theTolerance =  Margin(tolerance,tolerance); }
  void initLine(const PixelRecoLineRZ &line) { theLine = line; }
  void initLayer(const DetLayer *layer);
private:


  bool theBarrel, theForward;
  Range theDetRange, theDetSize;
  Margin theTolerance;
  PixelRecoLineRZ theLine;
};
#endif
