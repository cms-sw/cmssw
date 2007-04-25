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

 

  ///Produce PfRecTrack from a pair GsfTrack-Trajectory
  reco::PFRecTrack  producePFtrack(Trajectory * traj,
				   const reco::Track& track,
				   reco::PFRecTrack::AlgoType_t,
				   int index);

  ///Produce PfRecTrack from a pair Track-Trajectory
/*   reco::PFRecTrack  producePFtrack(Trajectory * traj, */
/* 				   reco::Track *ktrack, */
/* 				   reco::PFRecTrack::AlgoType_t, */
/* 				   int index); */

  reco::PFRecTrack  producePFtrack(Trajectory * traj,
				   const reco::TrackRef& trackref,
				   reco::PFRecTrack::AlgoType_t,
				   int index);

  ///Utility for getting the TSOS in all the surface defined in PFGeometry
  TrajectoryStateOnSurface getStateOnSurface(PFGeometry::Surface_t iSurf, 
					     const TrajectoryStateOnSurface& tsos, 
					     const Propagator* propagator, int& side);

  ///Surface corresponding to the expected mazimum shower of the electron 
  ReferenceCountingPointer<Surface> showerMaxSurface(float, bool,TSOS,int);

 private:

  /// Add to the PFRecTrack the points at ECAl , HCAL and beampipe
  void addPoints(); 

  ///Forward analytical Propagator
  const AnalyticalPropagator *fwdPropagator;

  ///Backward analytical Propagator
  const AnalyticalPropagator *bkwdPropagator;

  math::XYZTLorentzVector momClosest_;
  math::XYZPoint posClosest_;

  ///PFRecTrack returned in methods producePFtrackKf
  reco::PFRecTrack track_;

  ///Trajectory propagated to the surfaces of PFGeometry
  Trajectory *tj_;
};

#endif
