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

#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

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
    LogWarning("PFProducer")<<"invalid trajectory state on surface: "
			  <<" iSurf = "<<iSurf
			  <<" tan theta = "<<p.perp()/p.z()
			  <<" pz = "<<p.z()
			  <<endl;
  }
  
  return finalTSOS;
}





bool 
PFTrackTransformer::addPoints( reco::PFRecTrack& pftrack, 
			       const reco::Track& track,
			       const Trajectory& traj ) const {
  
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
     = PFCluster::getDepthCorrection(theOutParticle.momentum().mag(),
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

pair<float,float> PFTrackTransformer::showerDimension(float Eclus,
						      math::XYZPoint showerDirection, 
						      bool isBelowPS )const {
  double ecalShowerDepth     
    = PFCluster::getDepthCorrection(Eclus,
				    isBelowPS, 
				    false);
  showerDirection *= ecalShowerDepth/showerDirection.R();
  double rCyl = PFGeometry::innerRadius(PFGeometry::ECALBarrel) + 
    showerDirection.Rho();
  double zCyl = PFGeometry::innerZ(PFGeometry::ECALEndcap) + 
    fabs(showerDirection.Z());
  return make_pair(rCyl,zCyl);
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
