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
// $Id: ElectronPixelSeedGenerator.cc,v 1.15 2006/10/17 11:48:28 uberthon Exp $
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

ElectronPixelSeedGenerator::ElectronPixelSeedGenerator(float iephimin1, float iephimax1,
			                               float ipphimin1, float ipphimax1,
			                               float ipphimin2, float ipphimax2,
						       float izmin1, float izmax1,
						       float izmin2, float izmax2)
 : ephimin1(iephimin1), ephimax1(iephimax1), pphimin1(ipphimin1), pphimax1(ipphimax1), pphimin2(ipphimin2),	
   pphimax2(ipphimax2),zmin1(izmin1),zmax1(izmax1),zmin2(izmin2),zmax2(izmax2),
   myMatchEle(0), myMatchPos(0),
   theMode_(unknown), theUpdator(0), thePropagator(0), theMeasurementTracker(0), 
   theNavigationSchool(0), theSetup(0), pts_(0)
{}

ElectronPixelSeedGenerator::~ElectronPixelSeedGenerator() {

  delete theNavigationSchool;
  delete myMatchEle;
  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;

}


void ElectronPixelSeedGenerator::setupES(const edm::EventSetup& setup, const edm::ParameterSet &conf) {

  theSetup= &setup;

  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );

  theUpdator = new KFUpdator();
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 
  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));

   edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
  setup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  theMeasurementTracker = measurementTrackerHandle.product();

  myMatchEle->setES(&(*theMagField),theMeasurementTracker);
  myMatchPos->setES(&(*theMagField),theMeasurementTracker);

  //FIXME: to be reactivated
  //  moduleLabelBarrel_=conf.getParameter<string>("superClusterBarrelProducer");
  //  instanceNameBarrel_=conf.getParameter<string>("superClusterBarrelLabel");
  //CC@@
  //moduleLabelEndcap_=conf.getParameter<string>("superClusterEndcapProducer");
  //instanceNameEndcap_=conf.getParameter<string>("superClusterEndcapLabel");
}

void  ElectronPixelSeedGenerator::run(edm::Event& e, const edm::Handle<SuperClusterCollection> &clusters, ElectronPixelSeedCollection & out){

  theMeasurementTracker->update(e);
  
  NavigationSetter setter(*theNavigationSchool);

  //FIXME: to be reactivated
  // get input clusters 
  //  edm::Handle<SuperClusterCollection> bclusters;
  //  e.getByLabel(moduleLabelBarrel_,instanceNameBarrel_,bclusters);

  for  (unsigned int i=0;i<clusters->size();++i) {
    edm::Ref<SuperClusterCollection> theClusB(clusters,i);
    // Find the seeds
    recHits_.clear();
    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(theClusB,out) ;
  }

  if(theMode_==offline) LogDebug ("run") << "(offline)";
  
  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<clusters->size()
   <<", no. of ElectronPixelSeeds found  = " << out.size();
}

void ElectronPixelSeedGenerator::setup(bool off)
{

  if(theMode_==unknown)
    {
      // Instantiate the pixel hit matchers
      LogDebug("") << "ElectronPixelSeedGenerator, phi limits: " << ephimin1 << ", " << ephimax1 << ", "
		   << pphimin1 << ", " << pphimax1;
      myMatchEle = new PixelHitMatcher( ephimin1, ephimax1, pphimin2, pphimax2, zmin1, zmax1, zmin2, zmax2);
      myMatchPos = new PixelHitMatcher( pphimin1, pphimax1, pphimin2, pphimax2, zmin1, zmax1, zmin2, zmax2);
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
   LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] new supercluster with energy: " << clusterEnergy;
   LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] and position: " << clusterPos;
   
  PropagationDirection dir = alongMomentum;
   
  // is this an electron
  double aCharge=-1.;
   
  vector<pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits = 
    myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  float vertexZ = myMatchEle->getVertex();
  GlobalPoint eleVertex(0.,0.,vertexZ); 
 
  int isEle = 0;
  if (!elePixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] electron compatible hits found ";
    isEle = 1;
    vector<pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
     
    for (v = elePixelHits.begin(); v != elePixelHits.end(); v++) {
       
      (*v).first.invert();
      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,eleVertex);
      if (valid) {
        ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);
        result.push_back(s);
      }
    }
  }  
  aCharge=1.;  
  
  vector<pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits = 
    myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  vertexZ = myMatchPos->getVertex();
   
  GlobalPoint posVertex(0.,0.,vertexZ); 
  if (!posPixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] positron compatible hits found ";
    isEle == 1 ? isEle = 3 : isEle = 2;
    vector<pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
    for (v = posPixelHits.begin(); v != posPixelHits.end(); v++) {
      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
      if (valid) result.push_back(ElectronPixelSeed(seedCluster,*pts_,recHits_,dir));
    }
  } 

 return ;
}

bool ElectronPixelSeedGenerator::prepareElTrackSeed(ConstRecHitPointer innerhit,
						    ConstRecHitPointer outerhit,
						    const GlobalPoint& vertexPos)
{
  
  // debug prints
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] outer PixelHit   x,y,z "<<outerhit->globalPosition();

  recHits_.clear();
    
  SiPixelRecHit *hit;
  hit=new SiPixelRecHit(*(dynamic_cast <const SiPixelRecHit *> (innerhit->hit())));
  recHits_.push_back(hit);
  hit=new SiPixelRecHit(*(dynamic_cast <const SiPixelRecHit *> (outerhit->hit())));
  recHits_.push_back(hit);  
  

  typedef TrajectoryStateOnSurface     TSOS;
  // make a spiral
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) {
    return false;
  }
  FreeTrajectoryState fts = helix.stateAtVertex();
  TSOS propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid()) 
    return false;
  TSOS updatedState = theUpdator->update(propagatedState, *innerhit);
  
  TSOS propagatedState_out = thePropagator->propagate(fts,outerhit->det()->surface()) ;
  if (!propagatedState_out.isValid()) 
    return false;
  TSOS updatedState_out = theUpdator->update(propagatedState_out, *outerhit);
  // debug prints
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS Pt: "<<updatedState_out.globalMomentum().perp();
  pts_ =  transformer_.persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
  
}
