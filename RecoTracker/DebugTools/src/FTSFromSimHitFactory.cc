#include "RecoTracker/DebugTools/interface/FTSFromSimHitFactory.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include <algorithm>

FreeTrajectoryState FTSFromSimHitFactory::operator()( const PSimHit& hit, 
						      const GeomDetUnit& det,
						      const MagneticField& field) const
{
  GlobalVector momenta = det.toGlobal( hit.momentumAtEntry());
  TrackCharge ch = charge( hit.particleType());
  GlobalTrajectoryParameters param( det.toGlobal( hit.localPosition()), momenta, ch, &field);
  return FreeTrajectoryState( param);
}

TrackCharge FTSFromSimHitFactory::charge( int particleId) const
{
  if (std::abs( particleId) < 20) {
    // lepton 
    return TrackCharge( (particleId > 0) ? -1 : 1);
  }
  else {
    // only correct for stable mesoms and baryons
    return TrackCharge( (particleId > 0) ? 1 : -1);
  }
}
