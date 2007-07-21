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
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"



using namespace std;
using namespace reco;
using namespace edm;



PFTrackTransformer::PFTrackTransformer( const MagneticField * magField){
  LogInfo("PFTrackTransformer")<<"PFTrackTransformer built";

  fwdPropagator_=new AnalyticalPropagator(magField,alongMomentum);
  bkwdPropagator_=new AnalyticalPropagator(magField,oppositeToMomentum);
  maxShPropagator_=new StraightLinePropagator(magField);
  PFGeometry pfGeometry;
}



PFTrackTransformer::~PFTrackTransformer(){
  delete fwdPropagator_;
  delete bkwdPropagator_;
  delete maxShPropagator_;
}



TrajectoryStateOnSurface 
PFTrackTransformer::getStateOnSurface( PFGeometry::Surface_t iSurf, 
				       const TrajectoryStateOnSurface& tsos, 
				       const Propagator* propagator, int& side)
  const {

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




// PFRecTrack 
// PFTrackTransformer::producePFtrack(PFRecTrack& pftrack, 
// 				   Trajectory * traj,
// 				   const reco::TrackRef& trackref,
// 				   PFRecTrack::AlgoType_t algo,
// 				   int index){
  
// //   track_ =PFRecTrack( trackref->charge(), 
// // 		      algo,index, trackref );
// //   momClosest_= math::XYZTLorentzVector(trackref->px(), trackref->py(), 
// // 				       trackref->pz(), trackref->p());
// //   posClosest_=trackref->vertex();
//   tj_=traj;
  
//   addPoints(track, traj);
//   LogDebug("PFTrackTransformer")<<"Track "<< index <<"of algo "<<algo<<"transformed in PFTrack"; 
//   return track_;
// }


// PFRecTrack 
// PFTrackTransformer::producePFtrack(Trajectory * traj,
// 				   const reco::Track& track,
// 				   PFRecTrack::AlgoType_t algo,
// 				   int index){
// //   TrackRef dummyRef;

// //   track_ =PFRecTrack( track.charge(), 
// // 		      algo,index, dummyRef );
// //   momClosest_= math::XYZTLorentzVector(track.px(), track.py(), 
// // 				       track.pz(), track.p());
// //   posClosest_=track.vertex();
//   tj_=traj;
  
//   addPoints(track, traj);
//   LogDebug("PFTrackTransformer")<<"Track "<< index <<"of algo "<<algo<<"transformed in PFTrack"; 
//   return track_;
// }


bool 
PFTrackTransformer::addPoints( reco::PFRecTrack& pftrack, 
			       const reco::Track& track,
			       const Trajectory& traj ) const {
  
  LogDebug("PFTrackTransformer")<<"Trajectory propagation started";
  using namespace reco;
  
  // closest approach

  math::XYZTLorentzVector momClosest 
    = math::XYZTLorentzVector(track.px(), track.py(), 
			      track.pz(), track.p());
  math::XYZPoint posClosest = track.vertex();
  
  pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ClosestApproach,
				     posClosest,momClosest));
  // general info
  // directons, measurements,inner and outer state
  bool direction =(traj.direction() == alongMomentum);
  vector<TrajectoryMeasurement> measurements =traj.measurements();
  const TSOS innerTSOS= (direction) ?
    measurements[0].updatedState() : measurements[measurements.size() - 1].updatedState();
  TSOS outerTSOS= (direction) ?
    measurements[measurements.size() - 1].updatedState() : measurements[0].updatedState();

 
  //beam pipe
  int side=10000;
  if (posClosest.Rho() < PFGeometry::innerRadius(PFGeometry::BeamPipe)) {
    TSOS beamPipeTSOS = 
      getStateOnSurface(PFGeometry::BeamPipeWall, innerTSOS, 
			bkwdPropagator_, side);
    if(!beamPipeTSOS.isValid() ) return false;

    GlobalPoint v=beamPipeTSOS.globalPosition();
    GlobalVector p=beamPipeTSOS.globalMomentum();
    pftrack.addPoint(PFTrajectoryPoint(-1,
				       PFTrajectoryPoint::BeamPipeOrEndVertex,
				       math::XYZPoint(v.x(), v.y(), v.z()),
				       math::XYZTLorentzVector(p.x(),
							       p.y(),
							       p.z(),
							       p.mag())));
    //    LogDebug("PFProducer")<<"beam pipe point "<<endl;
  
  } else return false;


  //trajectory points
 
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

   //ECAL 
   int ecalSide = 100;
   TSOS ecalTSOS = 
     getStateOnSurface(PFGeometry::ECALInnerWall, outerTSOS, 
		       fwdPropagator_, ecalSide);
   if (!ecalTSOS.isValid()) return false; 

   GlobalPoint v=ecalTSOS.globalPosition();
   GlobalVector p=ecalTSOS.globalMomentum();
   pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALEntrance,
				      math::XYZPoint(v.x(), v.y(), v.z()),
				      math::XYZTLorentzVector(p.x(),
							      p.y(),
							      p.z(),
							      p.mag())));
   
   //preshower
   bool isBelowPS = false;
   if (v.perp()<PFGeometry::innerRadius(PFGeometry::ECALBarrel)){
     
     //layer 1
     TSOS ps1TSOS = 
       getStateOnSurface(PFGeometry::PS1Wall, outerTSOS, 
			 fwdPropagator_, side);     
     if( !ps1TSOS.isValid() ) return false;
     
     GlobalPoint v=ps1TSOS.globalPosition();
     GlobalVector p=ps1TSOS.globalMomentum();
     if (v.perp() >= PFGeometry::innerRadius(PFGeometry::PS1) &&
	 v.perp() <= PFGeometry::outerRadius(PFGeometry::PS1)) {
       isBelowPS = true;
       pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS1,
					  math::XYZPoint(v.x(), v.y(), v.z()),
					  math::XYZTLorentzVector(p.x(),
								  p.y(),
								  p.z(),
								  p.mag())));
     }
     else{ 
       PFTrajectoryPoint dummyPS1;
       pftrack.addPoint(dummyPS1); 
     }
     
     //layer 2
     TSOS ps2TSOS = 
       getStateOnSurface(PFGeometry::PS2Wall, outerTSOS, 
			 fwdPropagator_, side);     
     if( !ps2TSOS.isValid() ) return false;

     v=ps2TSOS.globalPosition();
     p=ps2TSOS.globalMomentum();
     if (v.perp() >= PFGeometry::innerRadius(PFGeometry::PS2) &&
	 v.perp() <= PFGeometry::outerRadius(PFGeometry::PS2)) {
       isBelowPS = true;
       pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::PS2,
					  math::XYZPoint(v.x(), v.y(), v.z()),
					  math::XYZTLorentzVector(p.x(),
								  p.y(),
								  p.z(),
								  p.mag())));
     }
     else{
       PFTrajectoryPoint dummyPS2;
       pftrack.addPoint(dummyPS2); 
     }
   } else{ // barrel, no preshower
     PFTrajectoryPoint dummyPS1;
     PFTrajectoryPoint dummyPS2;
     pftrack.addPoint(dummyPS1); 
     pftrack.addPoint(dummyPS2); 
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
       fwdPropagator_->propagate(ecalTSOS, *showerMaxWall);
//        maxShPropagator_->propagate(*(ecalTSOS.freeTrajectoryState()),
// 				   *showerMaxWall);
     if (!showerMaxTSOS.isValid()) return false; 

     v=showerMaxTSOS.globalPosition();
     p=showerMaxTSOS.globalMomentum();
     pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::ECALShowerMax,
					math::XYZPoint(v.x(), v.y(), v.z()),
					math::XYZTLorentzVector(p.x(),
								p.y(),
								p.z(),
								p.mag())));
   }
   
   //HCAL
   TSOS hcalTSOS = 
     getStateOnSurface(PFGeometry::HCALInnerWall, ecalTSOS, 
		       fwdPropagator_, side);
   if (!hcalTSOS.isValid() ) return false; 
   
   v=hcalTSOS.globalPosition();
   p=hcalTSOS.globalMomentum();
   pftrack.addPoint(PFTrajectoryPoint(-1,PFTrajectoryPoint::HCALEntrance,
				      math::XYZPoint(v.x(), v.y(), v.z()),
				      math::XYZTLorentzVector(p.x(),
							      p.y(),
							      p.z(),
							      p.mag())));
   

   return true;
}



ReferenceCountingPointer<Surface>
PFTrackTransformer::showerMaxSurface(float Eclus, bool isBelowPS, 
				     TSOS ecalTSOS,int side) const {
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
