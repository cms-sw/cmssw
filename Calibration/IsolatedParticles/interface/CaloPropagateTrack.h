#ifndef CalibrationIsolatedParticleseCCaloPropagateTrack_h
#define CalibrationIsolatedParticleseCCaloPropagateTrack_h

#include <cmath>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace spr{

  struct propagatedTrack {
    propagatedTrack() {ok=false;}
    bool                ok;
    math::XYZPoint      point;
    GlobalVector        direction;
  };

  propagatedTrack propagateTrackToECAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateECAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateECAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug=false);

  propagatedTrack propagateTrackToHCAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateHCAL(const reco::Track*, const MagneticField*, bool debug=false);
  std::pair<math::XYZPoint,bool> propagateHCAL(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, bool debug=false);

  propagatedTrack propagateCalo(const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, float zdist, float radius, float corner, bool debug=false);

}
#endif
