// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronPixelSeedGenerator.
// 
/**\class ElectronPixelSeedGenerator EgammaElectronAlgos/ElectronPixelSeedGenerator

 Description: Top algorithm producing ElectronPixelSeeds, ported from ORCA

 Implementation:
     future redesign...
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id$
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h" 
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h" 
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h" 
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"


ElectronPixelSeedGenerator::ElectronPixelSeedGenerator(float iephimin1,	
						 float iephimax1,	
						 float ipphimin1,	
						 float ipphimax1,	
						 float iphimin2,
						 float iphimax2,
						 float izmin1,	
						 float izmax1, 
						 float izmin2, 
						 float izmax2 
						 ) :
  ephimin1(iephimin1),
  ephimax1(iephimax1), 	
  pphimin1(ipphimin1), 	
  pphimax1(ipphimax1),
  phimin2(iphimin2),
  phimax2(iphimax2),
  zmin1(izmin1),
  zmax1(izmax1), 	  
  zmin2(izmin2),
  zmax2(izmax2),
  theMode_(unknown),
  theMeasurementTracker(0)
{
  
  //phi limits: 1st search
  //  SimpleConfigurable<float> ephimin_conf(-0.03,conf_name+":ePhiMin1"); 
  //  SimpleConfigurable<float> ephimax_conf(0.02,conf_name+":ePhiMax1"); 
  //  SimpleConfigurable<float> pphimin_conf(-0.02,conf_name+":pPhiMin1"); 
  //  SimpleConfigurable<float> pphimax_conf(0.03,conf_name+":pPhiMax1"); 

  //            2nd search
  //  SimpleConfigurable<float> phimin2_conf(-0.001,conf_name+":PhiMin2"); 
  //  SimpleConfigurable<float> phimax2_conf(0.001,conf_name+":PhiMax2"); 

  // z limits:
  //  SimpleConfigurable<float> zmin1_conf(-15.0,conf_name+":ZMin1"); 
  //  SimpleConfigurable<float> zmax1_conf(15.0,conf_name+":ZMax1"); 

  //SimpleConfigurable<float> zmin2_conf(-0.05,conf_name+":ZMin2"); 
  //  SimpleConfigurable<float> zmax2_conf(0.05,conf_name+":ZMax2"); 
  //UB FIXME
  //  pixrecHits_=new edm::OwnVector<SiPixelRecHit>;
  //  pixrecHits_=new std::vector<SiPixelRecHit>;

}

ElectronPixelSeedGenerator::~ElectronPixelSeedGenerator() {
  delete theMeasurementTracker;
  delete theNavigationSchool;
  delete myMatchEle;
  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;
}


void ElectronPixelSeedGenerator::setupES(const edm::EventSetup& setup, const edm::ParameterSet &conf) {
  // sets all what is dependent on this eventsetup
//   edm::ValidityInterval v1= setup.get<IdealMagneticFieldRecord>().validityInterval();
//   edm::ValidityInterval v2= setup.get<TrackerDigiGeometryRecord>().validityInterval();
//   edm::ValidityInterval v3= setup.get<TrackerRecoGeometryRecord>().validityInterval();
//   if (v1==vMag && v2==vTrackerDigi && v3==vTrackerReco) return;

   theSetup= &setup;
//   vMag=v1;
//   vTrackerDigi=v2;
//   vTrackerReco=v3;

  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );

  theUpdator = new KFUpdator();
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 

  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));
  theMeasurementTracker = new MeasurementTracker(setup,conf);

  myMatchEle->setES(&(*theMagField),theMeasurementTracker);
  myMatchPos->setES(&(*theMagField),theMeasurementTracker);
}

void  ElectronPixelSeedGenerator::run(const edm::Event& e, ElectronPixelSeedCollection & out){

  theMeasurementTracker->update(e);
  
  NavigationSetter setter(*theNavigationSchool);

  // get input clusters 
  edm::Handle<SuperClusterCollection> clusters;
  e.getByType(clusters);


  for  ( unsigned int i=0;i<clusters.product()->size();++i) {

    edm::Ref<SuperClusterCollection> theClus(clusters,i);
    // Find the seeds
    recHits_.clear();
    seedsFromThisCluster(theClus,out) ;
  }
  cout << "ElectronPixelSeedGenerator";
  if(theMode_==offline) cout << "(offline)";
  std::cout << ": For event "<<e.id()<<", nr of superclusters: "<<clusters.product()->size()<<", no. of ElectronPixelSeeds found = " << out.size() << std::endl;
  
}

void ElectronPixelSeedGenerator::setup(bool off)
{

   if(theMode_==unknown)
     {
      // Instantiate the pixel hit matchers
       //      cout << "ElectronPixelSeedGenerator, phi limits: " << ephimin1 << ", " << ephimax1 << ", "
       //	   << pphimin1 << ", " << pphimax1 << endl;
      myMatchEle = new PixelHitMatcher( ephimin1, ephimax1, phimin2, phimax2, zmin1, zmax1, zmin2, zmax2);
      myMatchPos = new PixelHitMatcher( pphimin1, pphimax1, phimin2, phimax2, zmin1, zmax1, zmin2, zmax2);
           if(off) theMode_=offline; else theMode_ = HLT;
        }
}
void ElectronPixelSeedGenerator::seedsFromThisCluster( edm::Ref<SuperClusterCollection> seedCluster, ElectronPixelSeedCollection& result)
{
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(),
			 seedCluster->position().y(), 
			 seedCluster->position().z());
  const GlobalPoint vertexPos(0.,0.,0.);
   
   
  PropagationDirection dir = alongMomentum;
   
  // is this an electron
  double aCharge=-1.;
   
  vector<pair<RecHitWithDist,TSiPixelRecHit> > elePixelHits = 
    myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  float vertexZ = myMatchEle->getVertex();
  GlobalPoint eleVertex(0.,0.,vertexZ); 
 
  int isEle = 0;
  if (!elePixelHits.empty() ) {
    isEle = 1;
    //     vector<pair<RecHitWithDist,RecHit> >::iterator v;
    vector<pair<RecHitWithDist,TSiPixelRecHit> >::iterator v;
     
    for (v = elePixelHits.begin(); v != elePixelHits.end(); v++) {
       
      //     try {  // 
      // change the sign of the phi distance for electrons
      (*v).first.invert();
      //UB FIXME: oseed?
      //      if (theMode_==offline) {
      //	 cache_.push_back(EPHLTElectronSeed(*v,seedCluster,eleVertex,vertexErr,dir,setup,"oseed")) ;
      // 	 result.push_back(EPHLTElectronSeed(*v,seedCluster,eleVertex,vertexErr,dir,setup,"oseed")) ;
      prepareElTrackSeed((*v).first.recHit(),(*v).second,eleVertex);
      //	 result.push_back(EPHLTElectronSeed(seedCluster,(*v).first.dPhi(),*pts_,recHits_,dir));
      ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);
      result.push_back(s);
      //      } else {
      //	 cache_.push_back(EPHLTElectronSeed(*v,seedCluster,eleVertex,vertexErr,dir,setup)) ;
      //	 result.push_back(EPHLTElectronSeed(*v,seedCluster,eleVertex,vertexErr,dir,setup)) ;
      //       }
    }
    //      catch( DetLogicError& err) {
    //        cout << "ElectronPixelSeedGenerator Warning: " << err.what() << endl;
			  
  }  
  aCharge=1.;  
  
  vector<pair<RecHitWithDist,TSiPixelRecHit> > posPixelHits = 
    myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  vertexZ = myMatchPos->getVertex();
   
  GlobalPoint posVertex(0.,0.,vertexZ); 
  if (!posPixelHits.empty() ) {
    isEle == 1 ? isEle = 3 : isEle = 2;
    vector<pair<RecHitWithDist,TSiPixelRecHit> >::iterator v;
    for (v = posPixelHits.begin(); v != posPixelHits.end(); v++) {
      //     try {
      //        if (theMode_==offline) {
      // 	 //	 cache_.push_back(EPHLTElectronSeed(*v,seedCluster,posVertex,vertexErr,dir,setup,"oseed")) ; 
      // 	 result.push_back(EPHLTElectronSeed(*v,seedCluster,posVertex,vertexErr,dir,setup,"oseed")) ; 
      //        } else {
      // 	 //	 cache_.push_back(EPHLTElectronSeed(*v,seedCluster,posVertex,vertexErr,dir,setup)) ; 
      // 	 result.push_back(EPHLTElectronSeed(*v,seedCluster,posVertex,vertexErr,dir,setup)) ; 
      prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
      result.push_back(ElectronPixelSeed(seedCluster,*pts_,recHits_,dir));
      //      }
      //      catch( DetLogicError& err) {
      //        cout << "ElectronPixelSeedGenerator Warning: " << err.what() << endl;
      //      }
    }
  } 

 return ;
}
void ElectronPixelSeedGenerator::prepareElTrackSeed(const TSiPixelRecHit outerhit,const TSiPixelRecHit innerhit, const GlobalPoint vertexPos) {

  // debug prints
  std::cout <<" outer PixelHit   x,y,z "<<outerhit.globalPosition()<<std::endl;
  std::cout <<" inner PixelHit   x,y,z "<<innerhit.globalPosition()<<std::endl;

  SiPixelRecHit *hit;
  hit=new SiPixelRecHit(*(dynamic_cast <SiPixelRecHit *> (outerhit.hit())));
  recHits_.push_back(hit);
  hit=new SiPixelRecHit(*(dynamic_cast <SiPixelRecHit *> (innerhit.hit())));
  recHits_.push_back(hit);

  typedef TrajectoryStateOnSurface     TSOS;
  // make a spiral
  FastHelix helix(outerhit.globalPosition(),innerhit.globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) {
    throw cms::Exception("DetLogicError")<<" prepareElTrackSeed: invalid helix";
  }
  FreeTrajectoryState fts = helix.stateAtVertex();
  TSOS propagatedState = thePropagator->propagate(fts,innerhit.det()->surface()) ;
  if (!propagatedState.isValid()) 
    throw cms::Exception("DetLogicError") <<"SeedFromConsecutiveHits propagation failed";
  TSOS updatedState = theUpdator->update(propagatedState, innerhit);
  
  TSOS propagatedState_out = thePropagator->propagate(fts,outerhit.det()->surface()) ;
  if (!propagatedState_out.isValid()) 
    throw cms::Exception("DetLogicError") <<"SeedFromConsecutiveHits propagation failed";
  TSOS updatedState_out = theUpdator->update(propagatedState_out, outerhit);
  // debug prints
  //  std::cout<<" final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum()<<std::endl;
  pts_ =  transformer.persistentState(updatedState_out, outerhit.geographicalId().rawId());
}
