#ifndef CalibrationIsolatedParticleseCCaloPropagateTrack_h
#define CalibrationIsolatedParticleseCCaloPropagateTrack_h

#include <cmath>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace spr{
  std::pair<math::XYZPoint,bool> propagateECAL( const reco::Track*, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateHCAL( const reco::Track*, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateCalo( const reco::Track*, const MagneticField*, float zdist, float radius, float corner ) ;

}
#endif
