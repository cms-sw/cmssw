//
// -*- C++ -*-
// Package:    PFTracking
// Class:      PFTrackTransformer
// 
// Original Author:  Michele Pioppi
// Other Author: Daniele Benedetti

#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// Add by Daniele
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "RecoParticleFlow/PFTracking/interface/PFGsfHelper.h"

using namespace std;
using namespace reco;
using namespace edm;



PFTrackTransformer::PFTrackTransformer(math::XYZVector B):B_(B){
  LogInfo("PFTrackTransformer")<<"PFTrackTransformer built";

  PFGeometry pfGeometry;
  onlyprop_=false;
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

  float PT= track.pt();
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
			   0.,0.,B_.z());

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
			   0.,0.,B_.z());
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
      unsigned int iid=measurements[iTraj].recHit()->det()->geographicalId().rawId();
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
     if (PT>5.)
       LogWarning("PFTrackTransformer")<<"KF TRACK "<<pftrack<< " PROPAGATION TO THE ECAL HAS FAILED";
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
   else{
     if (PT>5.)
       LogWarning("PFTrackTransformer")<<"KF TRACK "<<pftrack<< " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
     PFTrajectoryPoint dummyHCALentrance;
     pftrack.addPoint(dummyHCALentrance); 
   }

   //HCAL exit
   theOutParticle.propagateToHcalExit(false);
   if(theOutParticle.getSuccess()!=0)
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALExit,
					math::XYZPoint(theOutParticle.vertex()),
					math::XYZTLorentzVector(theOutParticle.momentum())));
   else{
     if (PT>5.)
       LogWarning("PFTrackTransformer")<<"KF TRACK "<<pftrack<< " PROPAGATION TO THE HCAL EXIT HAS FAILED";
     PFTrajectoryPoint dummyHCALexit;
     pftrack.addPoint(dummyHCALexit); 
   }
   
   return true;
}
bool 
PFTrackTransformer::addPointsAndBrems( reco::GsfPFRecTrack& pftrack, 
				       const reco::Track& track,
				       const Trajectory& traj,
				       const bool& GetMode) const {

  float PT= track.pt();
  // Trajectory for each trajectory point

  bool direction =(traj.direction() == alongMomentum);
  vector<TrajectoryMeasurement> measurements =traj.measurements();
  int iTrajFirst = (direction) ? 0 :  measurements.size() - 1;
  int increment = (direction) ? +1 : -1;
  int iTrajLast  =  (direction) ? int(measurements.size()) : -1;
  
  
  unsigned int iTrajPos = 0;
  for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
    
    GlobalPoint v=measurements[iTraj].updatedState().globalPosition();
    PFGsfHelper* PFGsf = new PFGsfHelper(measurements[iTraj]);
    //if (PFGsf->isValid()){ 
    bool ComputeMODE = GetMode;
    GlobalVector p = PFGsf->computeP(ComputeMODE);
    double DP = PFGsf->fittedDP();
    double SigmaDP =  PFGsf->sigmafittedDP();   
    unsigned int iid=measurements[iTraj].recHit()->det()->geographicalId().rawId();
    delete PFGsf;

    // --------------------------   Fill GSF Track ------------------------------------- 
    

    float pfmass= (pftrack.algoType()==reco::PFRecTrack::KF_ELCAND) ? 0.0005 : 0.139; 
    float ptot =  sqrt((p.x()*p.x())+(p.y()*p.y())+(p.z()*p.z()));
    float pfenergy=sqrt((pfmass*pfmass)+(ptot *ptot));

    if (iTraj == iTrajFirst) {

      math::XYZTLorentzVector momClosest 
	= math::XYZTLorentzVector(p.x(), p.y(), 
				  p.z(), ptot);
      math::XYZPoint posClosest = track.vertex();
      pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ClosestApproach,
					 posClosest,momClosest));
      
      BaseParticlePropagator theInnerParticle = 
	BaseParticlePropagator( 
			       RawParticle(XYZTLorentzVector(p.x(),
							     p.y(),
							     p.z(),
							     pfenergy),
					   XYZTLorentzVector(track.vertex().x(),
							     track.vertex().y(),
							     track.vertex().z(),
							     0.)),  //DANIELE Same thing v.x(),v.y(),v.()? 
			       0.,0.,B_.z());
      theInnerParticle.setCharge(track.charge());  

      //BEAMPIPE
      theInnerParticle.setPropagationConditions(PFGeometry::outerRadius(PFGeometry::BeamPipe), 
					   PFGeometry::outerZ(PFGeometry::BeamPipe), false);
      theInnerParticle.propagate();
      if(theInnerParticle.getSuccess()!=0)
	pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::BeamPipeOrEndVertex,
					   math::XYZPoint(theInnerParticle.vertex()),
					   math::XYZTLorentzVector(theInnerParticle.momentum())));
      else {
	PFTrajectoryPoint dummyMaxSh;
	pftrack.addPoint(dummyMaxSh); 
      }
      
      // First Point for the trajectory == Vertex ?? 
      pftrack.addPoint(PFTrajectoryPoint(iid,-1,
					 math::XYZPoint(v.x(), v.y(), v.z()),
					 math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
      
     
    }
    if (iTraj != iTrajFirst && iTraj != (abs(iTrajLast)-1)) {
      pftrack.addPoint(PFTrajectoryPoint(iid,-1,
					 math::XYZPoint(v.x(), v.y(), v.z()),
					 math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
      
 
    }
    if (iTraj == (abs(iTrajLast)-1)) {
      
      // Last Trajectory Meas
      pftrack.addPoint(PFTrajectoryPoint(iid,-1,
					 math::XYZPoint(v.x(), v.y(), v.z()),
					 math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));




      BaseParticlePropagator theOutParticle = 
	BaseParticlePropagator( 
			       RawParticle(XYZTLorentzVector(p.x(),
							     p.y(),
							     p.z(),
							     pfenergy),
					   XYZTLorentzVector(v.x(),
							     v.y(),
							     v.z(),
							     0.)), 
			       0.,0.,B_.z());
      theOutParticle.setCharge(track.charge());  
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
	if (PT>5.)
	  LogWarning("PFTrackTransformer")<<"GSF TRACK "<<pftrack<< " PROPAGATION TO THE ECAL HAS FAILED";
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
      else{
	if (PT>5.)
	  LogWarning("PFTrackTransformer")<<"GSF TRACK "<<pftrack<< " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
	PFTrajectoryPoint dummyHCALentrance;
	pftrack.addPoint(dummyHCALentrance); 
      }  
      
      //HCAL exit
      theOutParticle.propagateToHcalExit(false);
      if(theOutParticle.getSuccess()!=0)
	pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALExit,
					   math::XYZPoint(theOutParticle.vertex()),
					   math::XYZTLorentzVector(theOutParticle.momentum())));
      else{
	if (PT>5.)
	  LogWarning("PFTrackTransformer")<<"GSF TRACK "<<pftrack<< " PROPAGATION TO THE HCAL EXIT HAS FAILED";
	PFTrajectoryPoint dummyHCALexit;
	pftrack.addPoint(dummyHCALexit); 
      } 
      
    }

    // --------------------------   END GSF Track ------------------------------------- 
  
    // --------------------------   Fill Brem "Track" --------------------------------- 
    // Fill the brem for each traj point

    //check that the vertex of the brem is in the tracker volume
    if ((v.perp()>110) ||(fabs(v.z())>280)) continue;    
    unsigned int iTrajPoint =  iTrajPos + 2;
    if(iid%2 == 1) iTrajPoint = 99;

    PFBrem brem(DP,SigmaDP,iTrajPoint);


    GlobalVector p_gamma= p*(fabs(DP)/p.mag());   // Direction from the electron (tangent), DP without any sign!;
    float e_gamma = fabs(DP); // DP = pout-pin so could be negative
    BaseParticlePropagator theBremParticle = 
      BaseParticlePropagator( 
			     RawParticle(XYZTLorentzVector(p_gamma.x(),
							   p_gamma.y(),
							   p_gamma.z(),
							   e_gamma),
					 XYZTLorentzVector(v.x(),
							   v.y(),
							   v.z(),
							   0.)),
			     0.,0.,B_.z());
    int gamma_charge = 0;
    theBremParticle.setCharge(gamma_charge);  


    // add TrajectoryPoint for Brem, PS, ECAL, ECALShowMax, HCAL
    // Brem Entrance PS Layer1

    PFTrajectoryPoint dummyClosest;   // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyClosest); 

    
    PFTrajectoryPoint dummyBeamPipe;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyBeamPipe); 


    
    bool isBelowPS=false; 
    theBremParticle.propagateToPreshowerLayer1(false);
    if(theBremParticle.getSuccess()!=0)
      brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS1,
					 math::XYZPoint(theBremParticle.vertex()),
					 math::XYZTLorentzVector(theBremParticle.momentum())));
    else {
      PFTrajectoryPoint dummyPS1;
      brem.addPoint(dummyPS1); 
    }
    
    // Brem Entrance PS Layer 2

    theBremParticle.propagateToPreshowerLayer2(false);
    if(theBremParticle.getSuccess()!=0){
      brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS2,
					 math::XYZPoint(theBremParticle.vertex()),
					 math::XYZTLorentzVector(theBremParticle.momentum())));
      isBelowPS=true;
    }   else {
      PFTrajectoryPoint dummyPS2;
      brem.addPoint(dummyPS2); 
    }

   theBremParticle.propagateToEcalEntrance(false);

   if(theBremParticle.getSuccess()!=0){
     brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALEntrance,
					math::XYZPoint(theBremParticle.vertex()),
					math::XYZTLorentzVector(theBremParticle.momentum())));
   double ecalShowerDepth     
     = PFCluster::getDepthCorrection(theBremParticle.momentum().E(),
				     isBelowPS, 
				     false);

   math::XYZPoint meanShower=math::XYZPoint(theBremParticle.vertex())+
     math::XYZTLorentzVector(theBremParticle.momentum()).Vect().Unit()*ecalShowerDepth;
 
   brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALShowerMax,
				      meanShower,
				      math::XYZTLorentzVector(theBremParticle.momentum())));}
   else {
     if ((DP>5.) && ((DP/SigmaDP)>3))
       LogWarning("PFTrackTransformer")<<"BREM "<<brem<<" PROPAGATION TO THE ECAL HAS FAILED";
     PFTrajectoryPoint dummyECAL;
     brem.addPoint(dummyECAL); 
     PFTrajectoryPoint dummyMaxSh;
     brem.addPoint(dummyMaxSh); 
   }


 
   //HCAL entrance
   theBremParticle.propagateToHcalEntrance(false);
   if(theBremParticle.getSuccess()!=0)
     brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALEntrance,
					math::XYZPoint(theBremParticle.vertex()),
					math::XYZTLorentzVector(theBremParticle.momentum())));
   else{
     if ((DP>5.) && ((DP/SigmaDP)>3))
       LogWarning("PFTrackTransformer")<<"BREM "<<brem<<" PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
     PFTrajectoryPoint dummyHCALentrance;
     brem.addPoint(dummyHCALentrance); 
   }  

   //HCAL exit
   theBremParticle.propagateToHcalExit(false);
   if(theBremParticle.getSuccess()!=0)
     brem.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALExit,
					math::XYZPoint(theBremParticle.vertex()),
					math::XYZTLorentzVector(theBremParticle.momentum())));
   else{  
     if ((DP>5.) && ((DP/SigmaDP)>3))
       LogWarning("PFTrackTransformer")<<"BREM "<<brem<<" PROPAGATION TO THE HCAL EXIT HAS FAILED";
     PFTrajectoryPoint dummyHCALexit;
     brem.addPoint(dummyHCALexit); 
   }

   brem.calculatePositionREP();
   pftrack.addBrem(brem);
   iTrajPos++;
  }
  return true;
}
