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
// $Id: ElectronPixelSeedGenerator.cc,v 1.49 2008/04/08 16:39:15 uberthon Exp $
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

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include <vector>
#include <utility>
ElectronPixelSeedGenerator::ElectronPixelSeedGenerator(const edm::ParameterSet &pset)
  :   dynamicphiroad_(pset.getParameter<bool>("dynamicPhiRoad")),
      fromTrackerSeeds_(pset.getParameter<bool>("fromTrackerSeeds")),
      initialSeeds_(pset.getParameter<edm::InputTag>("initialSeeds")),
      lowPtThreshold_(pset.getParameter<double>("LowPtThreshold")),
      highPtThreshold_(pset.getParameter<double>("HighPtThreshold")),
      sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
      phimin2_(pset.getParameter<double>("PhiMin2")),      
      phimax2_(pset.getParameter<double>("PhiMax2")),
      deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
      deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
      deltaPhi2_(pset.getParameter<double>("DeltaPhi2")),
      myMatchEle(0), myMatchPos(0),
      thePropagator(0), theMeasurementTracker(0), 
      theSetup(0), pts_(0),
      cacheIDMagField_(0),cacheIDGeom_(0),cacheIDNavSchool_(0),cacheIDCkfComp_(0)
{ 
     // Instantiate the pixel hit matchers
  myMatchEle = new PixelHitMatcher( pset.getParameter<double>("ePhiMin1"), 
				    pset.getParameter<double>("ePhiMax1"),
				    pset.getParameter<double>("PhiMin2"),
				    pset.getParameter<double>("PhiMax2"),
				    pset.getParameter<double>("z2MinB"),
				    pset.getParameter<double>("z2MaxB"),
				    pset.getParameter<double>("r2MinF"),
				    pset.getParameter<double>("r2MaxF"),
				    pset.getParameter<double>("rMinI"),
				    pset.getParameter<double>("rMaxI"),
				    pset.getParameter<bool>("searchInTIDTEC"));

  myMatchPos = new PixelHitMatcher( pset.getParameter<double>("pPhiMin1"),
				    pset.getParameter<double>("pPhiMax1"),
				    pset.getParameter<double>("PhiMin2"),
				    pset.getParameter<double>("PhiMax2"),
				    pset.getParameter<double>("z2MinB"),
				    pset.getParameter<double>("z2MaxB"),
				    pset.getParameter<double>("r2MinF"),
				    pset.getParameter<double>("r2MaxF"),
				    pset.getParameter<double>("rMinI"),
				    pset.getParameter<double>("rMaxI"),
				    pset.getParameter<bool>("searchInTIDTEC"));

  theUpdator = new KFUpdator();
}

ElectronPixelSeedGenerator::~ElectronPixelSeedGenerator() {

  delete myMatchEle;
  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;
}

void ElectronPixelSeedGenerator::setupES(const edm::EventSetup& setup) {

  // get records if necessary (called once per event)
  bool tochange=false;

  if (cacheIDMagField_!=setup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    setup.get<IdealMagneticFieldRecord>().get(theMagField);
    cacheIDMagField_=setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    if (thePropagator) delete thePropagator;
    thePropagator = new PropagatorWithMaterial(alongMomentum,.000511,&(*theMagField)); 
    tochange=true;
  }
  if (cacheIDGeom_!=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier()) {
    setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
    cacheIDGeom_=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier();
  }

  if (cacheIDNavSchool_!=setup.get<NavigationSchoolRecord>().cacheIdentifier()) {
    edm::ESHandle<NavigationSchool> nav;
    setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
    cacheIDNavSchool_=setup.get<NavigationSchoolRecord>().cacheIdentifier();
  
    theNavigationSchool = nav.product();
  }				   

  if (cacheIDCkfComp_!=setup.get<CkfComponentsRecord>().cacheIdentifier()) {
    edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
    setup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
    cacheIDCkfComp_=setup.get<CkfComponentsRecord>().cacheIdentifier();
    theMeasurementTracker = measurementTrackerHandle.product();
    tochange=true;
  }
 
  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  if (cacheIDTrkGeom_!=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier()) {
    cacheIDTrkGeom_=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    setup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    tochange=true; //FIXME
  }

  if (tochange) {
    myMatchEle->setES(&(*theMagField),theMeasurementTracker,trackerGeometryHandle.product());
    myMatchPos->setES(&(*theMagField),theMeasurementTracker,trackerGeometryHandle.product());
  }

}

void  ElectronPixelSeedGenerator::run(edm::Event& e, const edm::EventSetup& setup, const reco::SuperClusterRefVector &sclRefs, reco::ElectronPixelSeedCollection & out){

  theSetup= &setup; 
  NavigationSetter theSetter(*theNavigationSchool);

  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_) e.getByLabel(initialSeeds_, theInitialSeedColl);
 
  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);

  // get its position
  BSPosition_ = recoBeamSpotHandle->position();
  double sigmaZ=recoBeamSpotHandle->sigmaZ();
  double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
  double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  zmin1_=BSPosition_.z()-3*sq;
  zmax1_=BSPosition_.z()+3*sq;

  theMeasurementTracker->update(e); 
  
 for  (unsigned int i=0;i<sclRefs.size();++i) {
    // Find the seeds
    recHits_.clear();

    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(sclRefs[i],out);
  }
  
  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters after filter: "<<sclRefs.size()
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

  myMatchEle->set1stLayerZRange(zmin1_,zmax1_);
  myMatchPos->set1stLayerZRange(zmin1_,zmax1_);
  
  if (dynamicphiroad_)
    {

      float clusterEnergyT = clusterEnergy*sin(seedCluster->position().theta()) ;

      float deltaPhi1 = 0.875/clusterEnergyT + 0.055; 
      if (clusterEnergyT < lowPtThreshold_) deltaPhi1= deltaPhi1Low_;
      if (clusterEnergyT > highPtThreshold_) deltaPhi1= deltaPhi1High_;

      float ephimin1 = -deltaPhi1*sizeWindowENeg_ ;
      float ephimax1 =  deltaPhi1*(1.-sizeWindowENeg_);
      float pphimin1 = -deltaPhi1*(1.-sizeWindowENeg_);
      float pphimax1 =  deltaPhi1*sizeWindowENeg_;

      float phimin2  = -deltaPhi2_/2. ;
      float phimax2  =  deltaPhi2_/2. ;

      myMatchEle->set1stLayer(ephimin1,ephimax1);
      myMatchPos->set1stLayer(pphimin1,pphimax1);
      myMatchEle->set2ndLayer(phimin2,phimax2);
      myMatchPos->set2ndLayer(phimin2,phimax2);

    }

  PropagationDirection dir = alongMomentum;
  
   // try electron
  double aCharge=-1.;
 
  if (!fromTrackerSeeds_) {
    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits = 
      myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
 
    float vertexZ = myMatchEle->getVertex();
    GlobalPoint eleVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);

    if (!elePixelHits.empty() ) {
      LogDebug("ElectronPixelSeedGenerator") << "seedsFromThisCluster: electron compatible hits found ";

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
  } else {
    std::vector<TrajectorySeed> elePixelSeeds=
      myMatchEle->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
      std::vector<TrajectorySeed>::iterator s;
      for (s = elePixelSeeds.begin(); s != elePixelSeeds.end(); s++) {
	  reco::ElectronPixelSeed seed(seedCluster,*s);
	  result.push_back(seed);
      }
  }

  // try positron
  aCharge=1.;  
  
  if (!fromTrackerSeeds_) {
    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits = 
      myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
 
    float vertexZ = myMatchPos->getVertex();
    GlobalPoint posVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);

    if (!posPixelHits.empty() ) {
      LogDebug("ElectronPixelSeedGenerator") << "seedsFromThisCluster: positron compatible hits found ";

      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
      for (v = posPixelHits.begin(); v != posPixelHits.end(); v++) {

	bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
	if (valid) {
	  reco::ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);	
	  result.push_back(s);
	  delete pts_;
	  pts_=0;
	}
      }
    }
  } else {
    std::vector<TrajectorySeed> posPixelSeeds=
      myMatchPos->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
    std::vector<TrajectorySeed>::iterator s;
    for (s = posPixelSeeds.begin(); s != posPixelSeeds.end(); s++) {
      reco::ElectronPixelSeed seed(seedCluster,*s);
      result.push_back(seed);
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
  
  SiPixelRecHit *pixhit=0;
  SiStripMatchedRecHit2D *striphit=0;
  const SiPixelRecHit* constpixhit = dynamic_cast <const SiPixelRecHit*> (innerhit->hit());
  if (constpixhit) {
    pixhit=new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit); 
  } else  return false;
  constpixhit =  dynamic_cast <const SiPixelRecHit *> (outerhit->hit());
  if (constpixhit) {
    pixhit=new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit); 
  } else {
    const SiStripMatchedRecHit2D * conststriphit=dynamic_cast <const SiStripMatchedRecHit2D *> (outerhit->hit());
    if (conststriphit) {
      striphit = new SiStripMatchedRecHit2D(*conststriphit);
      recHits_.push_back(striphit);   
    } else return false;
  }

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
  
  TSOS propagatedState_out = thePropagator->propagate(updatedState,outerhit->det()->surface()) ;
  if (!propagatedState_out.isValid()) 
    return false;
  TSOS updatedState_out = theUpdator->update(propagatedState_out, *outerhit);
  // debug prints
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS Pt: "<<updatedState_out.globalMomentum().perp();
  pts_ =  transformer_.persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
}
