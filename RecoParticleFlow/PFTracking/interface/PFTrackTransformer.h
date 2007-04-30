#ifndef PFTrackTransformer_H
#define PFTrackTransformer_H

#include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


/// \brief Abstract
/*!
\author Michele Pioppi
\date January 2007

 PFTrackTransformer is the class used to transform 
 pairs tracks-trajectories in pfrectracks that can be 
 used in the PFlow FW.
 Other utilities: 
 Get TSOS on surface defined in PFGEometry 
 Evaluate the surface corresponding to the maximum shower
*/

class MagneticField;
class Trajectory;
class AnalyticalPropagator;
class TrajectoryStateOnSurface;
class Propagator;

class PFTrackTransformer{

  typedef TrajectoryStateOnSurface TSOS;

 public:
  PFTrackTransformer(const MagneticField * magField);
  ~PFTrackTransformer();

 

/*   ///Produce PfRecTrack from a pair GsfTrack-Trajectory */
/*   reco::PFRecTrack  producePFTrack(reco::PFRecTrack& pftrack, */
/* 				   Trajectory * traj, */
/* 				   const reco::Track& track, */
/* 				   reco::PFRecTrack::AlgoType_t, */
/* 				   int index); */


/*   reco::PFRecTrack  producePFTrack(reco::PFRecTrack& pftrack, */
/* 				   Trajectory * traj, */
/* 				   const reco::TrackRef& trackref, */
/* 				   reco::PFRecTrack::AlgoType_t, */
/* 				   int index); */

  /// Add points to a PFTrack. return false if a TSOS is invalid
  bool addPoints(reco::PFRecTrack& pftrack, 
		 const reco::Track& track,
		 const Trajectory& traj ) const; 


  ///Utility for getting the TSOS in all the surface defined in PFGeometry
  TrajectoryStateOnSurface 
    getStateOnSurface(PFGeometry::Surface_t iSurf, 
		      const TrajectoryStateOnSurface& tsos, 
		      const Propagator* propagator, int& side) const;

  ///Surface corresponding to the expected mazimum shower of the electron 
  ReferenceCountingPointer<Surface> showerMaxSurface(float, 
						     bool,
						     TSOS,
						     int) const;

 private:

  ///Forward analytical Propagator
  const AnalyticalPropagator *fwdPropagator;

  ///Backward analytical Propagator
  const AnalyticalPropagator *bkwdPropagator;

/*   math::XYZTLorentzVector momClosest_; */
/*   math::XYZPoint posClosest_; */

  ///PFRecTrack returned in methods producePFtrackKf
  // reco::PFRecTrack track_;

  ///Trajectory propagated to the surfaces of PFGeometry
/*   Trajectory *tj_; */
};

#endif
