// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007
// $Id: JetTracksAssociationDRCalo.cc,v 1.4.2.1 2009/02/23 12:59:13 bainbrid Exp $

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

// -----------------------------------------------------------------------------
//
JetTracksAssociationDRCalo::JetTracksAssociationDRCalo( double fDr ) 
  : JetTracksAssociationDR(fDr),
    propagatedTracks_()
{;}

// -----------------------------------------------------------------------------
//
JetTracksAssociationDRCalo::~JetTracksAssociationDRCalo() 
{;}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRCalo::produce( Association* fAssociation, 
					  const Jets& fJets,
					  const Tracks& fTracks,
					  const TrackQuality& fQuality,
					  const MagneticField& fField,
					  const Propagator& fPropagator ) 
{
  JetRefs jets;
  createJetRefs( jets, fJets );
  TrackRefs tracks;
  createTrackRefs( tracks, fTracks, fQuality );
  produce( fAssociation, jets, tracks, fField, fPropagator );
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRCalo::produce( Association* fAssociation, 
					  const JetRefs& fJets,
					  const TrackRefs& fTracks,
					  const MagneticField& fField,
					  const Propagator& fPropagator ) 
{
  //clear();
  propagateTracks( fTracks, fField, fPropagator ); 
  associateTracksToJets( fAssociation, fJets, fTracks ); 
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRCalo::associateTracksToJet( reco::TrackRefVector& associated,
						       const reco::Jet& fJet,
						       const TrackRefs& fTracks ) 
{
  associated.clear();
  std::vector<ImpactPoint>::const_iterator ii = propagatedTracks_.begin();
  std::vector<ImpactPoint>::const_iterator jj = propagatedTracks_.end();
  for ( ; ii != jj; ++ii ) {
    double dR2 = deltaR2( fJet.eta(), fJet.phi(), ii->eta, ii->phi );
    if ( dR2 < mDeltaR2Threshold ) { associated.push_back( fTracks[ii->index] ); }
  }
}

// -----------------------------------------------------------------------------
//
void JetTracksAssociationDRCalo::propagateTracks( const TrackRefs& fTracks,
						  const MagneticField& fField,
						  const Propagator& fPropagator ) 
{
  propagatedTracks_.clear();
  propagatedTracks_.reserve( fTracks.size() );
  TrackRefs::const_iterator ii = fTracks.begin();
  TrackRefs::const_iterator jj = fTracks.end();
  for ( ; ii != jj; ++ii ) {
    GlobalPoint impact = JetTracksAssociationDRCalo::propagateTrackToCalo( **ii, fField, fPropagator );
    if ( impact.mag() > 0 ) { //@@ successful extrapolation
      ImpactPoint goodTrack;
      goodTrack.index = ii - fTracks.begin(); //@@ index
      goodTrack.eta = impact.eta();
      goodTrack.phi = impact.barePhi();
      propagatedTracks_.push_back( goodTrack );
    }
  }
}

// -----------------------------------------------------------------------------
//
math::XYZPoint JetTracksAssociationDRCalo::propagateTrackToCalorimeter( const reco::Track& fTrack,
									const MagneticField& fField,
									const Propagator& fPropagator )
{
  GlobalPoint result( JetTracksAssociationDRCalo::propagateTrackToCalo( fTrack, fField, fPropagator ) );
  return math::XYZPoint( result.x(), result.y(), result.z() ); 
}

// -----------------------------------------------------------------------------
//
GlobalPoint JetTracksAssociationDRCalo::propagateTrackToCalo( const reco::Track& fTrack,
							      const MagneticField& fField,
							      const Propagator& fPropagator )
{
  
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
					    *Cylinder::build (Surface::PositionType (0,0,0),
							      Surface::RotationType(),
							      rBarrel)
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

