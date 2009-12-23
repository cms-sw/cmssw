#ifndef CalibrationIsolatedParticleseCCaloPropagateTrack_h
#define CalibrationIsolatedParticleseCCaloPropagateTrack_h

#include <cmath>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace spr{
  math::XYZPoint propagateECAL( const reco::Track*, const MagneticField* ) ;
  math::XYZPoint propagateHCAL( const reco::Track*, const MagneticField* ) ;
  math::XYZPoint propagateCalo( const reco::Track*, const MagneticField*, float zdist, float radius, float corner ) ;

}
#endif
