#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

namespace spr{

  math::XYZPoint propagateECAL( const reco::Track *track, const MagneticField* bfield ) {
    return spr::propagateCalo (track, bfield, 319.2, 129.4, 1.479);
  }

  math::XYZPoint propagateHCAL( const reco::Track *track, const MagneticField* bfield ) {
    return spr::propagateCalo (track, bfield, 402.7, 180.7, 1.392);
  }

  math::XYZPoint propagateCalo( const reco::Track *track, const MagneticField* bField, float zdist, float radius, float corner ) {
    
    math::XYZPoint outerTrkPosition;
    
    GlobalPoint  tpVertex ( track->vx(), track->vy(), track->vz() );
    GlobalVector tpMomentum ( track->px(), track->py(), track->pz() );
    int tpCharge ( track->charge() );

    FreeTrajectoryState fts ( tpVertex, tpMomentum, tpCharge, bField);

    Plane::PlanePointer lendcap = Plane::build( Plane::PositionType (0, 0, -zdist), Plane::RotationType () );
    Plane::PlanePointer rendcap = Plane::build( Plane::PositionType (0, 0, zdist), Plane::RotationType () );
    
    Cylinder::CylinderPointer barrel = Cylinder::build( Cylinder::PositionType (0, 0, 0), Cylinder::RotationType (), radius);
  
    AnalyticalPropagator myAP (bField, alongMomentum, 2*M_PI);

    TrajectoryStateOnSurface tsose;
    if (track->eta() < 0) {
      tsose = myAP.propagate( fts, *lendcap);
    } else {
      tsose = myAP.propagate( fts, *rendcap);
    }

    TrajectoryStateOnSurface tsosb = myAP.propagate( fts, *barrel);

    if ( tsose.isValid() && tsosb.isValid() ) {
      float absEta = std::abs(tsosb.globalPosition().eta());
      if (absEta < corner)
	outerTrkPosition.SetXYZ( tsosb.globalPosition().x(), tsosb.globalPosition().y(), tsosb.globalPosition().z() );
      else
	outerTrkPosition.SetXYZ( tsose.globalPosition().x(), tsose.globalPosition().y(), tsose.globalPosition().z() );
    } else if ( tsose.isValid() ) {
      outerTrkPosition.SetXYZ( tsose.globalPosition().x(), tsose.globalPosition().y(), tsose.globalPosition().z() );
    } else if ( tsosb.isValid() ) {
      outerTrkPosition.SetXYZ( tsosb.globalPosition().x(), tsosb.globalPosition().y(), tsosb.globalPosition().z() );
    } else {
      outerTrkPosition.SetXYZ( -999., -999., -999. );
    }
    return outerTrkPosition;
  }

}
