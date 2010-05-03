#ifndef CalibrationIsolatedParticleseCCaloPropagateTrack_h
#define CalibrationIsolatedParticleseCCaloPropagateTrack_h

#include <cmath>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace spr{
  std::pair<math::XYZPoint,bool> propagateECAL( const reco::Track*, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateECAL( const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateHCAL( const reco::Track*, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateHCAL( const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* ) ;
  std::pair<math::XYZPoint,bool> propagateCalo( const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField*, float zdist, float radius, float corner ) ;

}
#endif
