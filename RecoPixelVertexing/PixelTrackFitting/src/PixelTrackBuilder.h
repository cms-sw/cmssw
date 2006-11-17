#ifndef PixelTrackBuilder_H
#define PixelTrackBuilder_H

#include <vector>
#include "DataFormats/TrackReco/interface/Track.h"
#include "Measurement1D.h"
class TrackingRecHit;
class MagneticField;

class PixelTrackBuilder {
public:
  reco::Track * build(
      const Measurement1D & pt,               // transverse momentu
      const Measurement1D & phi,              // direction at impact point
      const Measurement1D & cotTheta,         // cotangent of polar angle
      const Measurement1D & tip,              // closest approach in 2D
      const Measurement1D & zip,              // z at closest approach in 2D
      float chi2,                             // chi2 
      int   charge,                           // chi2
      const std::vector<const TrackingRecHit* >& hits,
      const MagneticField * mf) const;   
};

#endif
