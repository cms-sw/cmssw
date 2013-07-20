// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRCalo.cc,v 1.10 2012/12/26 14:25:08 innocent Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"


namespace {
  // basic geometry constants, imported from Geometry/HcalTowerAlgo/src/CaloTowerHardcodeGeometryLoader.cc
  const double rBarrel = 129.;
  const double zEndcap = 320.;
  const double zVF = 1100.;
  const double rEndcapMin = zEndcap * tan ( 2*atan (exp (-3.)));
  const double rVFMin = zEndcap * tan ( 2*atan (exp (-5.191)));

  struct ImpactPoint {
    unsigned index;
    double eta;
    double phi;
  };

  GlobalPoint propagateTrackToCalo (const reco::Track& fTrack,
				    const MagneticField& fField,
				    const Propagator& fPropagator)
  {
    GlobalPoint trackPosition (fTrack.vx(), fTrack.vy(), fTrack.vz()); // reference point
    GlobalVector trackMomentum (fTrack.px(), fTrack.py(), fTrack.pz()); // reference momentum
    if (fTrack.extra().isAvailable() ) { // use outer point information, if available
      trackPosition =  GlobalPoint (fTrack.outerX(), fTrack.outerY(), fTrack.outerZ());
      trackMomentum = GlobalVector (fTrack.outerPx(), fTrack.outerPy(), fTrack.outerPz());
    }
//     std::cout << "propagateTrackToCalo-> start propagating track"
// 	      << " x/y/z: " << trackPosition.x() << '/' << trackPosition.y() << '/' << trackPosition.z()
// 	      << ", pt/eta/phi: " << trackMomentum.perp() << '/' << trackMomentum.eta() << '/' << trackMomentum.barePhi()
// 	      << std::endl;
    GlobalTrajectoryParameters trackParams(trackPosition, trackMomentum, fTrack.charge(), &fField);
    FreeTrajectoryState trackState (trackParams);

    // first propagate to barrel
    TrajectoryStateOnSurface 
      propagatedInfo = fPropagator.propagate (trackState, 
					      *Cylinder::build (rBarrel, Surface::PositionType (0,0,0),
								Surface::RotationType())
					      );
    if (propagatedInfo.isValid()) {
      GlobalPoint result (propagatedInfo.globalPosition ());
      if (fabs (result.z()) < zEndcap) {
// 	std::cout << "propagateTrackToCalo-> propagated to barrel:"
// 		  << " x/y/z/r: " << result.x() << '/' << result.y() << '/' << result.z() << '/' << result.perp()
// 		  << std::endl;
	return result;
      }
    }
    
    // failed with barrel, try endcap
    double zTarget = trackMomentum.z() > 0 ? zEndcap : -zEndcap;
    propagatedInfo = fPropagator.propagate (trackState, 
					    *Plane::build( Surface::PositionType(0, 0, zTarget),
							   Surface::RotationType())
					    );
    if (propagatedInfo.isValid()) {
      GlobalPoint result (propagatedInfo.globalPosition ());
      if (fabs (result.perp()) > rEndcapMin) {
// 	std::cout << "propagateTrackToCalo-> propagated to endcap:"
// 		  << " x/y/z/r: " << result.x() << '/' << result.y() << '/' << result.z() << '/' << result.perp()
// 		  << std::endl;
	return result;
      }
    }
    // failed with endcap, try VF
    zTarget = trackMomentum.z() > 0 ? zVF : -zVF;
    propagatedInfo = fPropagator.propagate (trackState, 
					    *Plane::build( Surface::PositionType(0, 0, zTarget),
							   Surface::RotationType())
					    );
    if (propagatedInfo.isValid()) {
      GlobalPoint result (propagatedInfo.globalPosition ());
      if (fabs (result.perp()) > rVFMin) {
// 	std::cout << "propagateTrackToCalo-> propagated to VF:"
// 		  << " x/y/z/r: " << result.x() << '/' << result.y() << '/' << result.z() << '/' << result.perp()
// 		  << std::endl;
	return result;
      }
    }
    // no luck
//     std::cout << "propagateTrackToCalo-> failed to propagate track to calorimeter" << std::endl;
    return GlobalPoint (0, 0, 0);
  }
}

JetTracksAssociationDRCalo::JetTracksAssociationDRCalo (double fDr) 
: mDeltaR2Threshold (fDr*fDr)
{}

void JetTracksAssociationDRCalo::produce (reco::JetTracksAssociation::Container* fAssociation, 
					  const std::vector <edm::RefToBase<reco::Jet> >& fJets,
					  const std::vector <reco::TrackRef>& fTracks,
					  const MagneticField& fField,
					  const Propagator& fPropagator) const 
{
  // cache track parameters
  std::vector<ImpactPoint> impacts;
  for (unsigned t = 0; t < fTracks.size(); ++t) {
    GlobalPoint impact = propagateTrackToCalo (*(fTracks[t]), fField, fPropagator);
    if (impact.mag () > 0) { // successful extrapolation
      ImpactPoint goodTrack;
      goodTrack.index = t;
      goodTrack.eta = impact.eta ();
      goodTrack.phi = impact.barePhi();
      impacts.push_back (goodTrack);
    }
  }
  
  for (unsigned j = 0; j < fJets.size(); ++j) {
    reco::TrackRefVector assoTracks;
    const reco::Jet* jet = &*(fJets[j]); 
    double jetEta = jet->eta();
    double jetPhi = jet->phi();
    for (unsigned t = 0; t < impacts.size(); ++t) {
      double dR2 = deltaR2 (jetEta, jetPhi, impacts[t].eta, impacts[t].phi);
      if (dR2 < mDeltaR2Threshold)  assoTracks.push_back (fTracks[impacts[t].index]);
    }
    reco::JetTracksAssociation::setValue (fAssociation, fJets[j], assoTracks);
  }
}

math::XYZPoint JetTracksAssociationDRCalo::propagateTrackToCalorimeter (const reco::Track& fTrack,
								   const MagneticField& fField,
								   const Propagator& fPropagator)
{
  GlobalPoint result (propagateTrackToCalo (fTrack, fField, fPropagator));
  return math::XYZPoint (result.x(), result.y(), result.z()); 
}
