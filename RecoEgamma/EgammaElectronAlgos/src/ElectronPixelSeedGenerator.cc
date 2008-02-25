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
// $Id: ElectronPixelSeedGenerator.cc,v 1.35 2008/02/13 13:27:56 uberthon Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h" 

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h" 
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h" 
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include <vector>
#include <utility>

ElectronPixelSeedGenerator::ElectronPixelSeedGenerator(float iephimin1, float iephimax1,
			                               float ipphimin1, float ipphimax1,
			                               float ipphimin2, float ipphimax2,
						       //						       float izmin1, float izmax1,
						       float izmin2, float izmax2,
                                                       bool idynamicphiroad, double SCEtCut)
 : ephimin1(iephimin1), ephimax1(iephimax1), pphimin1(ipphimin1), pphimax1(ipphimax1), pphimin2(ipphimin2),	
   pphimax2(ipphimax2),
   zmin2(izmin2),zmax2(izmax2),dynamicphiroad(idynamicphiroad),SCEtCut_(SCEtCut),
   myMatchEle(0), myMatchPos(0),theUpdator(0), thePropagator(0), theMeasurementTracker(0), 
   theNavigationSchool(0), theSetup(0), pts_(0)
{
      // Instantiate the pixel hit matchers
      LogDebug("") << "ElectronPixelSeedGenerator, phi limits: " << ephimin1 << ", " << ephimax1 << ", "
		   << pphimin1 << ", " << pphimax1;
      myMatchEle = new PixelHitMatcher( ephimin1, ephimax1, pphimin2, pphimax2, zmin2, zmax2);
      myMatchPos = new PixelHitMatcher( pphimin1, pphimax1, pphimin2, pphimax2, zmin2, zmax2);
}

ElectronPixelSeedGenerator::~ElectronPixelSeedGenerator() {

  delete myMatchEle;
  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;

}


void ElectronPixelSeedGenerator::setupES(const edm::EventSetup& setup) {
  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );

  edm::ESHandle<NavigationSchool> nav;
  setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavigationSchool = nav.product();
  NavigationSetter setter(*theNavigationSchool);

  edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
  setup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  theMeasurementTracker = measurementTrackerHandle.product();

  if (theUpdator) delete theUpdator;
  theUpdator = new KFUpdator();
  if (thePropagator) delete thePropagator;
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 
  myMatchEle->setES(&(*theMagField),theMeasurementTracker); 
  myMatchPos->setES(&(*theMagField),theMeasurementTracker);
}

void  ElectronPixelSeedGenerator::run(edm::Event& e, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection> &clusters, reco::ElectronPixelSeedCollection & out){

  //Getting the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);

  // gets its position
  BSPosition_ = recoBeamSpotHandle->position();
  double sigmaZ=recoBeamSpotHandle->sigmaZ();
  double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
  double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  zmin1=BSPosition_.z()-3*sq;
  zmax1=BSPosition_.z()+3*sq;

  theSetup= &setup; 

  theMeasurementTracker->updatePixels(e);

  for  (unsigned int i=0;i<clusters->size();++i) {
    edm::Ref<reco::SuperClusterCollection> theClusB(clusters,i);
    // Find the seeds
    recHits_.clear();
    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    if (theClusB->energy()/cosh(theClusB->eta())>SCEtCut_)    seedsFromThisCluster(theClusB,out) ;
  }

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<clusters->size()
   <<", no. of ElectronPixelSeeds found  = " << out.size();
}

void ElectronPixelSeedGenerator::seedsFromThisCluster( edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection& result)
{
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(),
			 seedCluster->position().y(), 
			 seedCluster->position().z());

  const GlobalPoint vertexPos(BSPosition_.x(),BSPosition_.y(),BSPosition_.z());
  LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] new supercluster with energy: " << clusterEnergy;
  LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] and position: " << clusterPos;

  myMatchEle->set1stLayerZRange(zmin1,zmax1);
  myMatchPos->set1stLayerZRange(zmin1,zmax1);
  
  //Here change the deltaPhi window of the first pixel layer in function of the seed pT
  if (dynamicphiroad)
    {
      float clusterEnergyT = clusterEnergy*sin(seedCluster->position().theta()) ;

      float deltaPhi1 = 1.4/clusterEnergyT ;
      float deltaPhi2 = 0.07/clusterEnergyT ;
      float ephimin1 = -deltaPhi1*0.625 ;
      float ephimax1 =  deltaPhi1*0.375 ;
      float pphimin1 = -deltaPhi1*0.375 ;
      float pphimax1 =  deltaPhi1*0.625 ;
      float phimin2  = -deltaPhi2*0.5 ;
      float phimax2  =  deltaPhi2*0.5 ;

      if (clusterEnergyT < 5)
	{
	  ephimin1 = -0.280*0.625 ;
	  ephimax1 =  0.280*0.375 ;
	  pphimin1 = -0.280*0.375 ;
	  pphimax1 =  0.280*0.625 ;
	  phimin2  = -0.007 ;
	  phimin2  =  0.007 ;
	}

      if (clusterEnergyT > 35)
	{
	  ephimin1 = -0.040*0.625 ;
	  ephimax1 =  0.040*0.375 ;
	  pphimin1 = -0.040*0.375 ;
	  pphimax1 =  0.040*0.625 ;
	  phimin2  = -0.001 ;
	  phimax2  =  0.001 ;
	}


      myMatchEle->set1stLayer(ephimin1,ephimax1);
      myMatchPos->set1stLayer(pphimin1,pphimax1);
      myMatchEle->set2ndLayer(phimin2,phimax2);
      myMatchPos->set2ndLayer(phimin2,phimax2);

    }
  PropagationDirection dir = alongMomentum;
   
  // is this an electron
  double aCharge=-1.;
   
  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits = 
    myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  float vertexZ = myMatchEle->getVertex();
  GlobalPoint eleVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);
  int isEle = 0;
  if (!elePixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] electron compatible hits found ";
    isEle = 1;
    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
     
    for (v = elePixelHits.begin(); v != elePixelHits.end(); v++) {
       
      (*v).first.invert();
      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,eleVertex);
      if (valid) {
        reco::ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);
        result.push_back(s);
	delete pts_;
	pts_=0;
      }
    }
  }  
  aCharge=1.;  
  
  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits = 
    myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
  vertexZ = myMatchPos->getVertex();
   
  GlobalPoint posVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);
  if (!posPixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] positron compatible hits found ";
    isEle == 1 ? isEle = 3 : isEle = 2;
    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
    for (v = posPixelHits.begin(); v != posPixelHits.end(); v++) {
      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
      if (valid) {
	result.push_back(reco::ElectronPixelSeed(seedCluster,*pts_,recHits_,dir));
	delete pts_;
	pts_=0;
      }
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

  pts_=0;
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
