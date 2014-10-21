#ifndef PFTrackTransformer_H
#define PFTrackTransformer_H

#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"



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


class Trajectory;
class PFTrackTransformer{



 public:
  PFTrackTransformer(const math::XYZVector&);
  ~PFTrackTransformer();


  /// Add points to a PFTrack. return false if a TSOS is invalid
  bool addPoints(reco::PFRecTrack& pftrack, 
		 const reco::Track& track,
		 const Trajectory& traj,
		 bool msgwarning = true) const; 
  
  bool addPointsAndBrems(reco::GsfPFRecTrack& pftrack, 
			 const reco::Track& track,
			 const Trajectory& traj,
			 const bool& GetMode) const; 
  
  bool addPointsAndBrems(reco::GsfPFRecTrack& pftrack, 
			 const reco::GsfTrack& track,
			 const MultiTrajectoryStateTransform& mtjstate) const; 

  void OnlyProp(){
    onlyprop_=true;
  }
  bool  onlyprop_;
  
 private:
  ///B field
   math::XYZVector B_;
   const MultiTrajectoryStateMode *mtsMode_;
   PFGeometry pfGeometry_;
};

#endif
