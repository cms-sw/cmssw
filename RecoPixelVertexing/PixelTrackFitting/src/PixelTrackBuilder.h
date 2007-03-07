#ifndef PixelTrackBuilder_H
#define PixelTrackBuilder_H

#include <vector>
#include <string>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
class TrackingRecHit;
class MagneticField;
class TrajectoryStateOnSurface;
class FreeTrajectoryState;

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

private:
  std::string print(const reco::Track & track) const; 
  std::string print(const TrajectoryStateOnSurface & state) const;
  std::string print( const Measurement1D & pt,
    const Measurement1D & phi,
    const Measurement1D & cotTheta,
    const Measurement1D & tip,
    const Measurement1D & zip,
    float chi2,
    int   charge) const;

  void checkState(const TrajectoryStateOnSurface & state, const MagneticField* mf) const;

};

#endif
