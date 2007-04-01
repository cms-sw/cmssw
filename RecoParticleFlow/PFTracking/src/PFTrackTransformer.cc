// -*- C++ -*-
//
// Package:    PFTracking
// Class:      PFTrackTransformer
// 
// Original Author:  Michele Pioppi
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"



using namespace std;
using namespace reco;
using namespace edm;
PFTrackTransformer::PFTrackTransformer( const MagneticField * magField){
  LogInfo("PFTrackTransformer")<<"PFTrackTransformer built";

  fwdPropagator=new AnalyticalPropagator(magField,alongMomentum);
  bkwdPropagator=new AnalyticalPropagator(magField,oppositeToMomentum);

  PFGeometry pfGeometry;
}
PFTrackTransformer::~PFTrackTransformer(){
  delete fwdPropagator;
  delete bkwdPropagator;
}
TrajectoryStateOnSurface 
PFTrackTransformer::getStateOnSurface( PFGeometry::Surface_t iSurf, 
				       const TrajectoryStateOnSurface& tsos, 
				       const Propagator* propagator, int& side) {
  GlobalVector p = tsos.globalParameters().momentum();
  TSOS finalTSOS;
  side = -100;
  if (fabs(p.perp()/p.z()) > PFGeometry::tanTh(iSurf)) {
    finalTSOS = propagator->propagate(tsos, PFGeometry::barrelBound(iSurf));
    side = 0;
    if (!finalTSOS.isValid()) {
      if (p.z() > 0.) {
	finalTSOS = propagator->propagate(tsos, PFGeometry::positiveEndcapDisk(iSurf));
	side = 1;
      } else {
	finalTSOS = propagator->propagate(tsos, PFGeometry::negativeEndcapDisk(iSurf));
	side = -1;
      }
    }
  } else if (p.z() > 0.) {
    finalTSOS = propagator->propagate(tsos, PFGeometry::positiveEndcapDisk(iSurf));
    side = 1;
    if (!finalTSOS.isValid()) {
      finalTSOS = propagator->propagate(tsos, PFGeometry::barrelBound(iSurf));
      side = 0;
    }
  } else {
    finalTSOS = propagator->propagate(tsos, PFGeometry::negativeEndcapDisk(iSurf));
    side = -1;
    if (!finalTSOS.isValid()) {
      finalTSOS = propagator->propagate(tsos, PFGeometry::barrelBound(iSurf));
      side = 0;
    }
  }
  
  if( !finalTSOS.isValid() ) {
    LogError("PFProducer")<<"invalid trajectory state on surface: "
			  <<" iSurf = "<<iSurf
			  <<" tan theta = "<<p.perp()/p.z()
			  <<" pz = "<<p.z()
			  <<endl;
  }
  
  return finalTSOS;
}


PFRecTrack 
PFTrackTransformer::producePFtrackKf(Trajectory * traj,
				     GsfTrack *gtrack,
				     PFRecTrack::AlgoType_t algo,
				     int index){
  track_ =PFRecTrack(gtrack->charge(), 
		    algo,index);
  momClosest_= math::XYZTLorentzVector(gtrack->px(), gtrack->py(), 
				      gtrack->pz(), gtrack->p());
  posClosest_=gtrack->vertex();
  tj_=traj;

  addPoints();
  LogDebug("PFTrackTransformer")<<"Track "<< index <<"of algo "<<algo<<"transformed in PFTrack"; 
  return track_;
 
}

PFRecTrack 
PFTrackTransformer::producePFtrackKf(Trajectory * traj,
				     Track *ktrack,
				     PFRecTrack::AlgoType_t algo,
				     int index){
  track_ =PFRecTrack(ktrack->charge(), 
		    algo,index);
  momClosest_= math::XYZTLorentzVector(ktrack->px(), ktrack->py(), 
				      ktrack->pz(), ktrack->p());
  posClosest_=ktrack->vertex();
  tj_=traj;

  addPoints();
  LogDebug("PFTrackTransformer")<<"Track "<< index <<"of algo "<<algo<<"transformed in PFTrack"; 
  return track_;
 
}

void 
PFTrackTransformer::addPoints(){
  LogDebug("PFTrackTransformer")<<"Trajectory propagation started";
  using namespace reco;
  //beam
  track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ClosestApproach,
  				   posClosest_,momClosest_));
  // general info
  // directons, measurements,inner and outer state
  bool direction =(tj_->direction() == alongMomentum);
  vector<TrajectoryMeasurement> measurements =tj_->measurements();
  const TSOS innerTSOS= (direction) ?
    measurements[0].updatedState() : measurements[measurements.size() - 1].updatedState();
  TSOS outerTSOS= (direction) ?
    measurements[measurements.size() - 1].updatedState() : measurements[0].updatedState();

 
  //beam pipe
  int side=10000;
  if (posClosest_.Rho() < PFGeometry::innerRadius(PFGeometry::BeamPipe)) {
    TSOS beamPipeTSOS = 
      getStateOnSurface(PFGeometry::BeamPipeWall, innerTSOS, 
			bkwdPropagator, side);
    if(beamPipeTSOS.isValid() ){
      GlobalPoint v=beamPipeTSOS.globalPosition();
      GlobalVector p=beamPipeTSOS.globalMomentum();
      track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::BeamPipe,
				       math::XYZPoint(v.x(), v.y(), v.z()),
				       math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
      //    LogDebug("PFProducer")<<"beam pipe point "<<endl;
    }
  }


  //trajectory points
 
  int iTrajFirst = (direction) ? 0 :  measurements.size() - 1;
  int increment = (direction) ? +1 : -1;
  int iTrajLast  =  (direction) ? int(measurements.size()) : -1;
 

   for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
     GlobalPoint v=measurements[iTraj].updatedState().globalPosition();
     GlobalVector p=measurements[iTraj].updatedState().globalMomentum();
     uint iid=measurements[iTraj].recHit()->det()->geographicalId().rawId();
     track_.addPoint(PFTrajectoryPoint(iid,-1,
				      math::XYZPoint(v.x(), v.y(), v.z()),
				      math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
   }
   //ECAL 
   int ecalSide = 100;
   TSOS ecalTSOS = 
     getStateOnSurface(PFGeometry::ECALInnerWall, outerTSOS, 
		       fwdPropagator, ecalSide);
   if (ecalTSOS.isValid()){
     GlobalPoint v=ecalTSOS.globalPosition();
     GlobalVector p=ecalTSOS.globalMomentum();
     track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALEntrance,
				      math::XYZPoint(v.x(), v.y(), v.z()),
				      math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));

     //preshower
     bool isBelowPS = false;
     if (v.perp()<PFGeometry::innerRadius(PFGeometry::ECALBarrel)){


       //layer 1
       TSOS ps1TSOS = 
	 getStateOnSurface(PFGeometry::PS1Wall, outerTSOS, 
			   fwdPropagator, side);     
       if( ps1TSOS.isValid() ){
	 GlobalPoint v=ps1TSOS.globalPosition();
	 GlobalVector p=ps1TSOS.globalMomentum();
	 if (v.perp() >= PFGeometry::innerRadius(PFGeometry::PS1) &&
	     v.perp() <= PFGeometry::outerRadius(PFGeometry::PS1)) {
	   isBelowPS = true;
	   track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS1,
					    math::XYZPoint(v.x(), v.y(), v.z()),
					    math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
	 }
	 else{ 
	   PFTrajectoryPoint dummyPS1;
	   track_.addPoint(dummyPS1); 
	 }
       }
       //layer 2
       TSOS ps2TSOS = 
	 getStateOnSurface(PFGeometry::PS2Wall, outerTSOS, 
			   fwdPropagator, side);     
       if( ps2TSOS.isValid() ){
	 GlobalPoint v=ps2TSOS.globalPosition();
	 GlobalVector p=ps2TSOS.globalMomentum();
	 if (v.perp() >= PFGeometry::innerRadius(PFGeometry::PS2) &&
	     v.perp() <= PFGeometry::outerRadius(PFGeometry::PS2)) {
	   isBelowPS = true;
	   track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS2,
					    math::XYZPoint(v.x(), v.y(), v.z()),
					    math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
	 }
	 else{
	   PFTrajectoryPoint dummyPS2;
	   track_.addPoint(dummyPS2); 
	 }
       }



     } else{
       PFTrajectoryPoint dummyPS1;
       PFTrajectoryPoint dummyPS2;
       track_.addPoint(dummyPS1); 
       track_.addPoint(dummyPS2); 
     }
     // Propage track to ECAL shower max TODO
     // Be careful : the following formula are only valid for electrons !
     // Michele
     // Clusters energy replaced by track momentum

     ReferenceCountingPointer<Surface> showerMaxWall=
       showerMaxSurface(ecalTSOS.globalMomentum().mag(),isBelowPS,
			ecalTSOS,side);
     if(&(*showerMaxWall)!=0){
       TSOS showerMaxTSOS = 
	 fwdPropagator->propagate(ecalTSOS, *showerMaxWall);
     
       if (showerMaxTSOS.isValid()){
	 GlobalPoint v=showerMaxTSOS.globalPosition();
	 GlobalVector p=showerMaxTSOS.globalMomentum();
	 track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALShowerMax,
					  math::XYZPoint(v.x(), v.y(), v.z()),
					  math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
       }
     }
     
     //HCAL
     TSOS hcalTSOS = 
       getStateOnSurface(PFGeometry::HCALInnerWall, ecalTSOS, 
			 fwdPropagator, side);
     if (hcalTSOS.isValid()){
       GlobalPoint v=hcalTSOS.globalPosition();
       GlobalVector p=hcalTSOS.globalMomentum();
       track_.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALEntrance,
					math::XYZPoint(v.x(), v.y(), v.z()),
					math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.mag())));
     }
     
   }
}



ReferenceCountingPointer<Surface>
PFTrackTransformer::showerMaxSurface(float Eclus, bool isBelowPS, 
				     TSOS ecalTSOS,int side){
  double ecalShowerDepth     
      = PFCluster::getDepthCorrection(Eclus,
				      isBelowPS, 
				      false);
  math::XYZPoint showerDirection(ecalTSOS.globalMomentum().x(), 
				 ecalTSOS.globalMomentum().y(), 
				 ecalTSOS.globalMomentum().z());
  showerDirection *= ecalShowerDepth/showerDirection.R();
  double rCyl = PFGeometry::innerRadius(PFGeometry::ECALBarrel) + 
    showerDirection.Rho();
  double zCyl = PFGeometry::innerZ(PFGeometry::ECALEndcap) + 
    fabs(showerDirection.Z());
  

  ReferenceCountingPointer<Surface> showerMaxWall;
  const float epsilon = 0.001; // should not matter at all
  switch (side) {
  case 0: 
    showerMaxWall 
      = ReferenceCountingPointer
      <Surface>(new BoundCylinder(GlobalPoint(0.,0.,0.), 
				  TkRotation<float>(), 
				  SimpleCylinderBounds(rCyl, rCyl, -zCyl, zCyl))); 
    break;
  case +1: 
    showerMaxWall 
      = ReferenceCountingPointer
      <Surface>( new BoundPlane(Surface::PositionType(0,0,zCyl), 
				TkRotation<float>(), 
				SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
    break;
  case -1: 
    showerMaxWall 
      = ReferenceCountingPointer
      <Surface>(new BoundPlane(Surface::PositionType(0,0,-zCyl), 
			       TkRotation<float>(), 
			       SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
       break;
  }
  return  showerMaxWall;
}
