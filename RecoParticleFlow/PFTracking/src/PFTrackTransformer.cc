//
// -*- C++ -*-
// Package:    PFTracking
// Class:      PFTrackTransformer
// 
// Original Author:  Michele Pioppi
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

using namespace std;
using namespace reco;
using namespace edm;



PFTrackTransformer::PFTrackTransformer(){
  LogInfo("PFTrackTransformer")<<"PFTrackTransformer built";

  PFGeometry pfGeometry;
  onlyprop_=true;
}

PFTrackTransformer::~PFTrackTransformer(){

}









bool 
PFTrackTransformer::addPoints( reco::PFRecTrack& pftrack, 
			       const reco::Track& track,
			       const Trajectory& traj) const {
  
  LogDebug("PFTrackTransformer")<<"Trajectory propagation started";
  using namespace reco;
  using namespace std;


  float pfmass= (pftrack.algoType()==reco::PFRecTrack::KF_ELCAND) ? 0.0005 : 0.139; 
  float pfenergy=sqrt((pfmass*pfmass)+(track.p()*track.p()));
   // closest approach
  BaseParticlePropagator theParticle = 
    BaseParticlePropagator( 
			   RawParticle(XYZTLorentzVector(track.px(),
							 track.py(),
							 track.pz(),
							 pfenergy),
				       XYZTLorentzVector(track.vertex().x(),
							 track.vertex().y(),
							 track.vertex().z(),
							 0.)),
			   0.,0.,4.);
  theParticle.setCharge(track.charge());
  float pfoutenergy=sqrt((pfmass*pfmass)+track.outerMomentum().Mag2());
  BaseParticlePropagator theOutParticle = 
    BaseParticlePropagator( 
			   RawParticle(XYZTLorentzVector(track.outerMomentum().x(),
							 track.outerMomentum().y(),
							 track.outerMomentum().z(),
							 pfoutenergy),
				       XYZTLorentzVector(track.outerPosition().x(),
							 track.outerPosition().y(),
							 track.outerPosition().z(),
							 0.)),
			   0.,0.,4.);
  theOutParticle.setCharge(track.charge());
  
  
  math::XYZTLorentzVector momClosest 
    = math::XYZTLorentzVector(track.px(), track.py(), 
			      track.pz(), track.p());
  math::XYZPoint posClosest = track.vertex();
  
  pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ClosestApproach,
				     posClosest,momClosest));
  
  
  //BEAMPIPE
  theParticle.setPropagationConditions(PFGeometry::outerRadius(PFGeometry::BeamPipe), 
				       PFGeometry::outerZ(PFGeometry::BeamPipe), false);
  theParticle.propagate();
  if(theParticle.getSuccess()!=0)
    pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::BeamPipeOrEndVertex,
				       math::XYZPoint(theParticle.vertex()),
				       math::XYZTLorentzVector(theParticle.momentum())));
  else {
    PFTrajectoryPoint dummyMaxSh;
    pftrack.addPoint(dummyMaxSh); 
  }
  


  //trajectory points

  if (!onlyprop_){
    bool direction =(traj.direction() == alongMomentum);
    vector<TrajectoryMeasurement> measurements =traj.measurements();
    int iTrajFirst = (direction) ? 0 :  measurements.size() - 1;
    int increment = (direction) ? +1 : -1;
    int iTrajLast  =  (direction) ? int(measurements.size()) : -1;
    

    for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
      GlobalPoint v=measurements[iTraj].updatedState().globalPosition();
      GlobalVector p=measurements[iTraj].updatedState().globalMomentum();
      uint iid=measurements[iTraj].recHit()->det()->geographicalId().rawId();
      pftrack.addPoint(PFTrajectoryPoint(iid,-1,
					 math::XYZPoint(v.x(), v.y(), v.z()),
					 math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
    }
  }
  
   bool isBelowPS=false; 
   theOutParticle.propagateToPreshowerLayer1(false);
   if(theOutParticle.getSuccess()!=0)
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS1,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));
   else {
     PFTrajectoryPoint dummyPS1;
     pftrack.addPoint(dummyPS1); 
   }
   

   theOutParticle.propagateToPreshowerLayer2(false);
   if(theOutParticle.getSuccess()!=0){
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS2,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));
     isBelowPS=true;
   }   else {
     PFTrajectoryPoint dummyPS2;
     pftrack.addPoint(dummyPS2); 
   }

   theOutParticle.propagateToEcalEntrance(false);

   if(theOutParticle.getSuccess()!=0){
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALEntrance,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));
   double ecalShowerDepth     
     = PFCluster::getDepthCorrection(theOutParticle.momentum().E(),
				     isBelowPS, 
				     false);

   math::XYZPoint meanShower=math::XYZPoint(theOutParticle.vertex())+
     math::XYZTLorentzVector(theOutParticle.momentum()).Vect().Unit()*ecalShowerDepth;
 
   pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALShowerMax,
				      meanShower,
				      math::XYZTLorentzVector(theOutParticle.momentum())));}
   else {
     PFTrajectoryPoint dummyECAL;
     pftrack.addPoint(dummyECAL); 
     PFTrajectoryPoint dummyMaxSh;
     pftrack.addPoint(dummyMaxSh); 
   }


 
   //HCAL entrance
   theOutParticle.propagateToHcalEntrance(false);
   if(theOutParticle.getSuccess()!=0)
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALEntrance,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));
 

   //HCAL exit
   theOutParticle.propagateToHcalExit(false);
   if(theOutParticle.getSuccess()!=0)
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALExit,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));

   
   return true;
}
