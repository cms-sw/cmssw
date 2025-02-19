#ifndef ThirdHitRZPrediction_H
#define ThirdHitRZPrediction_H

/** predicts a range in r-z for the position of third hit.
 *  the predicted reange is defined by the template argument, which
 *  is a straight line extrapolation/interpolation if PixelRecoLineRZ is used.
 */

#include <algorithm>

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPredictionBase.h"

namespace helper {
  // Helper template to define how the ThirdHitRZPrediction interfaces
  // the prediction implementation
  template<class Propagator>
  class ThirdHitRZPredictionTraits {
  public:
    static inline float transform(const Propagator &propagator, bool barrel, float rOrZ);
  };

  // generic: defaults to calling zAtR or rAtZ
  // template can be specialized for individual classes if different
  // are required
  template<class Propagator>
  inline float ThirdHitRZPredictionTraits<Propagator>::transform(
  	const Propagator &propagator, bool barrel, float rOrZ)
  { return barrel ? propagator.zAtR(rOrZ) : propagator.rAtZ(rOrZ); }
}

template<class Propagator>
class ThirdHitRZPrediction : public ThirdHitRZPredictionBase {
public:
  typedef helper::ThirdHitRZPredictionTraits<Propagator> traits;

  ThirdHitRZPrediction() : ThirdHitRZPredictionBase(), thePropagator(0) {}
  ThirdHitRZPrediction(const Propagator *propagator, float tolerance, const DetLayer* layer = 0) :
      ThirdHitRZPredictionBase(tolerance, layer), thePropagator(propagator) {}

  Range operator()(const DetLayer *layer = 0);
  inline Range operator()(float rORz) const { return (*this)(rORz, *thePropagator); }
  Range operator()(float rORz, const Propagator &propagator) const;

  void initPropagator(const Propagator *propagator) { thePropagator = propagator; }

private:
  const Propagator *thePropagator;
};

template<class Propagator>
typename ThirdHitRZPrediction<Propagator>::Range
ThirdHitRZPrediction<Propagator>::operator()(const DetLayer *layer)
{
  if (layer) initLayer(layer);
  if (!theBarrel && !theForward) return Range(0., 0.);
  float v1 = traits::transform(*thePropagator, theBarrel, theDetRange.min());
  float v2 = traits::transform(*thePropagator, theBarrel, theDetRange.max());
  if (v1 > v2) std::swap(v1, v2);
  return Range(v1 - theTolerance.left(), v2 + theTolerance.right());
}

template<class Propagator>
typename ThirdHitRZPrediction<Propagator>::Range
ThirdHitRZPrediction<Propagator>::operator()(float rOrZ, const Propagator &propagator) const
{
  float v = traits::transform(propagator, theBarrel, rOrZ);
  return Range(v - theTolerance.left(), v + theTolerance.right());
}

#endif
