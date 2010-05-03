#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

namespace spr{

  std::pair<math::XYZPoint,bool> propagateECAL( const reco::Track *track, const MagneticField* bfield ) {    
    GlobalPoint  vertex ( track->vx(), track->vy(), track->vz() );
    GlobalVector momentum ( track->px(), track->py(), track->pz() );
    int charge ( track->charge() );
    return spr::propagateECAL (vertex, momentum, charge, bfield);
  }

  std::pair<math::XYZPoint,bool> propagateECAL( const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield ) {    
    return spr::propagateCalo (vertex, momentum, charge, bfield, 319.2, 129.4, 1.479);
  }

  std::pair<math::XYZPoint,bool> propagateHCAL( const reco::Track *track, const MagneticField* bfield ) {
    GlobalPoint  vertex ( track->vx(), track->vy(), track->vz() );
    GlobalVector momentum ( track->px(), track->py(), track->pz() );
    int charge ( track->charge() );
    return spr::propagateHCAL (vertex, momentum, charge, bfield);
  }

  std::pair<math::XYZPoint,bool> propagateHCAL( const GlobalPoint& vertex, const GlobalVector& momentum, int charge, const MagneticField* bfield ) {
    return spr::propagateCalo (vertex, momentum, charge, bfield, 402.7, 180.7, 1.392);
  }

  std::pair<math::XYZPoint,bool> propagateCalo( const GlobalPoint& tpVertex, const GlobalVector& tpMomentum, int tpCharge, const MagneticField* bField, float zdist, float radius, float corner ) {
    
    math::XYZPoint outerTrkPosition;
    FreeTrajectoryState fts ( tpVertex, tpMomentum, tpCharge, bField);

    Plane::PlanePointer lendcap = Plane::build( Plane::PositionType (0, 0, -zdist), Plane::RotationType () );
    Plane::PlanePointer rendcap = Plane::build( Plane::PositionType (0, 0,  zdist), Plane::RotationType () );
    
    Cylinder::CylinderPointer barrel = Cylinder::build( Cylinder::PositionType (0, 0, 0), Cylinder::RotationType (), radius);
  
    AnalyticalPropagator myAP (bField, alongMomentum, 2*M_PI);

    TrajectoryStateOnSurface tsose;
    if (tpMomentum.eta() < 0) {
      tsose = myAP.propagate( fts, *lendcap);
    } else {
      tsose = myAP.propagate( fts, *rendcap);
    }

    TrajectoryStateOnSurface tsosb = myAP.propagate( fts, *barrel);

    bool ok=true;
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
      ok = false;
    }
    return std::pair<math::XYZPoint,bool>(outerTrkPosition,ok);
  }

}
