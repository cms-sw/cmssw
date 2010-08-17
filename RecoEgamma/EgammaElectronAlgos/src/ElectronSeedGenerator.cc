// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronSeedGenerator.
//
/**\class ElectronSeedGenerator EgammaElectronAlgos/ElectronSeedGenerator

 Description: Top algorithm producing ElectronSeeds, ported from ORCA

 Implementation:
     future redesign...
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//

#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
//#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//#include "DataFormats/Common/interface/Handle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include <vector>
#include <utility>

ElectronSeedGenerator::ElectronSeedGenerator(const edm::ParameterSet &pset)
 : dynamicphiroad_(pset.getParameter<bool>("dynamicPhiRoad")),
   fromTrackerSeeds_(pset.getParameter<bool>("fromTrackerSeeds")),
   useRecoVertex_(false),
   verticesTag_("offlinePrimaryVerticesWithBS"),
   beamSpotTag_("offlineBeamSpot"),
   lowPtThreshold_(pset.getParameter<double>("LowPtThreshold")),
   highPtThreshold_(pset.getParameter<double>("HighPtThreshold")),
   nSigmasDeltaZ1_(pset.getParameter<double>("nSigmasDeltaZ1")),
   deltaZ1WithVertex_(0.5),
   sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
   deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
   deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
   myMatchEle(0), myMatchPos(0),
   thePropagator(0),
   theMeasurementTracker(0),
   theSetup(0), pts_(0),
   cacheIDMagField_(0),/*cacheIDGeom_(0),*/cacheIDNavSchool_(0),cacheIDCkfComp_(0),cacheIDTrkGeom_(0)
 {
  // use of a theMeasurementTrackerName
  if (pset.exists("measurementTrackerName"))
   { theMeasurementTrackerName = pset.getParameter<std::string>("measurementTrackerName") ; }

  // use of reco vertex
  if (pset.exists("useRecoVertex"))
   { useRecoVertex_ = pset.getParameter<bool>("useRecoVertex") ; }
  if (pset.exists("vertices"))
   { verticesTag_ = pset.getParameter<edm::InputTag>("vertices") ; }
  if (pset.exists("deltaZ1WithVertex"))
   { deltaZ1WithVertex_ = pset.getParameter<double>("deltaZ1WithVertex") ; }

  // new beamSpot tag
  if (pset.exists("beamSpot"))
   { beamSpotTag_ = pset.getParameter<edm::InputTag>("beamSpot") ; }

  // new B/F configurables
  if (pset.exists("DeltaPhi2"))
   { deltaPhi2B_ = deltaPhi2F_ = pset.getParameter<double>("DeltaPhi2") ; }
  else
   {
    deltaPhi2B_ = pset.getParameter<double>("DeltaPhi2B") ;
    deltaPhi2F_ = pset.getParameter<double>("DeltaPhi2F") ;
   }
  if (pset.exists("PhiMin2"))
   { phiMin2B_ = phiMin2F_ = pset.getParameter<double>("PhiMin2") ; }
  else
   {
    phiMin2B_ = pset.getParameter<double>("PhiMin2B") ;
    phiMin2F_ = pset.getParameter<double>("PhiMin2F") ;
   }
  if (pset.exists("PhiMax2"))
   { phiMax2B_ = phiMax2F_ = pset.getParameter<double>("PhiMax2") ; }
  else
   {
    phiMax2B_ = pset.getParameter<double>("PhiMax2B") ;
    phiMax2F_ = pset.getParameter<double>("PhiMax2F") ;
   }

  // Instantiate the pixel hit matchers
  myMatchEle = new PixelHitMatcher
   ( pset.getParameter<double>("ePhiMin1"),
		 pset.getParameter<double>("ePhiMax1"),
		 phiMin2B_, phiMax2B_, phiMin2F_, phiMax2F_,
     pset.getParameter<double>("z2MinB"),
     pset.getParameter<double>("z2MaxB"),
     pset.getParameter<double>("r2MinF"),
     pset.getParameter<double>("r2MaxF"),
     pset.getParameter<double>("rMinI"),
     pset.getParameter<double>("rMaxI"),
     pset.getParameter<bool>("searchInTIDTEC") ) ;

  myMatchPos = new PixelHitMatcher
   ( pset.getParameter<double>("pPhiMin1"),
		 pset.getParameter<double>("pPhiMax1"),
		 phiMin2B_, phiMax2B_, phiMin2F_, phiMax2F_,
     pset.getParameter<double>("z2MinB"),
     pset.getParameter<double>("z2MaxB"),
     pset.getParameter<double>("r2MinF"),
     pset.getParameter<double>("r2MaxF"),
     pset.getParameter<double>("rMinI"),
     pset.getParameter<double>("rMaxI"),
     pset.getParameter<bool>("searchInTIDTEC") ) ;

  theUpdator = new KFUpdator() ;
 }

ElectronSeedGenerator::~ElectronSeedGenerator()
 {
  delete myMatchEle ;
  delete myMatchPos ;
  delete thePropagator ;
  delete theUpdator ;
 }

void ElectronSeedGenerator::setupES(const edm::EventSetup& setup) {

  // get records if necessary (called once per event)
  bool tochange=false;

  if (cacheIDMagField_!=setup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    setup.get<IdealMagneticFieldRecord>().get(theMagField);
    cacheIDMagField_=setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    if (thePropagator) delete thePropagator;
    thePropagator = new PropagatorWithMaterial(alongMomentum,.000511,&(*theMagField));
    tochange=true;
  }

  if (!fromTrackerSeeds_ && cacheIDCkfComp_!=setup.get<CkfComponentsRecord>().cacheIdentifier()) {
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    setup.get<CkfComponentsRecord>().get(theMeasurementTrackerName,measurementTrackerHandle);
    cacheIDCkfComp_=setup.get<CkfComponentsRecord>().cacheIdentifier();
    theMeasurementTracker = measurementTrackerHandle.product();
    tochange=true;
  }

  //edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  if (cacheIDTrkGeom_!=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier()) {
    cacheIDTrkGeom_=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    setup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    tochange=true; //FIXME
  }

  if (tochange) {
    myMatchEle->setES(&(*theMagField),theMeasurementTracker,theTrackerGeometry.product());
    myMatchPos->setES(&(*theMagField),theMeasurementTracker,theTrackerGeometry.product());
  }

  if (cacheIDNavSchool_!=setup.get<NavigationSchoolRecord>().cacheIdentifier()) {
    edm::ESHandle<NavigationSchool> nav;
    setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
    cacheIDNavSchool_=setup.get<NavigationSchoolRecord>().cacheIdentifier();
    theNavigationSchool = nav.product();
  }

//  if (cacheIDGeom_!=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier()) {
//    setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
//    cacheIDGeom_=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier();
//  }

}

void  ElectronSeedGenerator::run
 ( edm::Event & e, const edm::EventSetup & setup,
   const reco::SuperClusterRefVector & sclRefs, const std::vector<float> & hoe1s, const std::vector<float> & hoe2s,
   TrajectorySeedCollection * seeds, reco::ElectronSeedCollection & out )
 {
  theInitialSeedColl=seeds;

  theSetup= &setup;
  NavigationSetter theSetter(*theNavigationSchool);

  // get initial TrajectorySeeds if necessary
  //  if (fromTrackerSeeds_) e.getByLabel(initialSeeds_, theInitialSeedColl);

  // get the beamspot from the Event:
  //e.getByType(theBeamSpot);
  e.getByLabel(beamSpotTag_,theBeamSpot);

  // if required get the vertices
  if (useRecoVertex_) e.getByLabel(verticesTag_,theVertices);

  if (!fromTrackerSeeds_)
   { theMeasurementTracker->update(e) ; }

  for  (unsigned int i=0;i<sclRefs.size();++i) {
    // Find the seeds
    recHits_.clear();

    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(sclRefs[i],hoe1s[i],hoe2s[i],out);
  }

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters after filter: "<<sclRefs.size()
   <<", no. of ElectronSeeds found  = " << out.size();
}

void ElectronSeedGenerator::seedsFromThisCluster
( edm::Ref<reco::SuperClusterCollection> seedCluster,
  float hoe1, float hoe2,
  reco::ElectronSeedCollection & result )
{
  float clusterEnergy = seedCluster->energy() ;
  GlobalPoint clusterPos
    ( seedCluster->position().x(),
      seedCluster->position().y(),
      seedCluster->position().z() ) ;
  LogDebug("") << "[ElectronSeedGenerator::seedsFromThisCluster] new supercluster with energy: " << clusterEnergy ;
  LogDebug("") << "[ElectronSeedGenerator::seedsFromThisCluster] and position: " << clusterPos ;

  if (dynamicphiroad_)
   {
    float clusterEnergyT = clusterEnergy / cosh( EleRelPoint(clusterPos,theBeamSpot->position()).eta() ) ;

    float deltaPhi1 = 0.875/clusterEnergyT + 0.055;
    if (clusterEnergyT < lowPtThreshold_) deltaPhi1= deltaPhi1Low_;
    if (clusterEnergyT > highPtThreshold_) deltaPhi1= deltaPhi1High_;

    float ephimin1 = -deltaPhi1*sizeWindowENeg_ ;
    float ephimax1 =  deltaPhi1*(1.-sizeWindowENeg_);
    float pphimin1 = -deltaPhi1*(1.-sizeWindowENeg_);
    float pphimax1 =  deltaPhi1*sizeWindowENeg_;

    float phimin2B  = -deltaPhi2B_/2. ;
    float phimax2B  =  deltaPhi2B_/2. ;
    float phimin2F  = -deltaPhi2F_/2. ;
    float phimax2F  =  deltaPhi2F_/2. ;


    myMatchEle->set1stLayer(ephimin1,ephimax1);
    myMatchPos->set1stLayer(pphimin1,pphimax1);
    myMatchEle->set2ndLayer(phimin2B,phimax2B, phimin2F,phimax2F);
    myMatchPos->set2ndLayer(phimin2B,phimax2B, phimin2F,phimax2F);
   }

  PropagationDirection dir = alongMomentum;

  if (!useRecoVertex_)
   { // here use the beam spot position
    double sigmaZ=theBeamSpot->sigmaZ();
    double sigmaZ0Error=theBeamSpot->sigmaZ0Error();
    double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
    zmin1_=theBeamSpot->position().z()-nSigmasDeltaZ1_*sq;
    zmax1_=theBeamSpot->position().z()+nSigmasDeltaZ1_*sq;

    GlobalPoint vertexPos ;
    ele_convert(theBeamSpot->position(),vertexPos) ;

    myMatchEle->set1stLayerZRange(zmin1_,zmax1_);
    myMatchPos->set1stLayerZRange(zmin1_,zmax1_);

    // try electron
    double aCharge=-1.;

    if (!fromTrackerSeeds_)
     {
      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits
       = myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);

      float vertexZ = myMatchEle->getVertex();
      GlobalPoint eleVertex(theBeamSpot->position().x(),theBeamSpot->position().y(),vertexZ);

      if (!elePixelHits.empty() )
       {
        LogDebug("ElectronSeedGenerator") << "seedsFromThisCluster: electron compatible hits found ";

        std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;

        for (v = elePixelHits.begin(); v != elePixelHits.end(); v++)
         {
          (*v).first.invert();
          bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,eleVertex);
          if (valid)
           {
            reco::ElectronSeed s(*pts_,recHits_,dir) ;
            s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
            result.push_back(s);
            delete pts_;
            pts_=0;
           }
         }
       }
     }
    else
     {
      std::vector<SeedWithInfo> elePixelSeeds
       = myMatchEle->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
      std::vector<SeedWithInfo>::iterator s;
      for (s = elePixelSeeds.begin(); s != elePixelSeeds.end(); s++)
       {
        reco::ElectronSeed seed(s->seed()) ;
        reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
        seed.setCaloCluster(caloCluster,s->subDet2(),s->dRz2(),s->dPhi2(),s->subDet1(),s->dRz1(),s->dPhi1(),hoe1,hoe2) ;
        result.push_back(seed) ;
       }
     }

    // try positron
    aCharge=1.;

    if (!fromTrackerSeeds_)
     {
      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits
       = myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);

      float vertexZ = myMatchPos->getVertex() ;
      GlobalPoint posVertex(theBeamSpot->position().x(),theBeamSpot->position().y(),vertexZ) ;

      if (!posPixelHits.empty() )
       {
	      LogDebug("ElectronSeedGenerator") << "seedsFromThisCluster: positron compatible hits found ";

        std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
        for (v = posPixelHits.begin(); v != posPixelHits.end(); v++)
         {
          bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
          if (valid)
           {
            reco::ElectronSeed s(*pts_,recHits_,dir) ;
            s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
            result.push_back(s);
            delete pts_;
            pts_=0;
           }
         }
       }
     }
    else
     {
      std::vector<SeedWithInfo> posPixelSeeds
       = myMatchPos->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
      std::vector<SeedWithInfo>::iterator s ;
      for (s = posPixelSeeds.begin(); s != posPixelSeeds.end(); s++)
       {
  	    reco::ElectronSeed seed(s->seed()) ;
        reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
        seed.setCaloCluster(caloCluster,s->subDet2(),s->dRz2(),s->dPhi2(),s->subDet1(),s->dRz1(),s->dPhi1(),hoe1,hoe2) ;
        result.push_back(seed);
       }
     }
   }
  else
   { // here we use the reco vertices
    const  std::vector<reco::Vertex> * vtxCollection = theVertices.product() ;
    std::vector<reco::Vertex>::const_iterator vtxIter ;
    for (vtxIter = vtxCollection->begin(); vtxIter != vtxCollection->end() ; vtxIter++)
     {
      GlobalPoint vertexPos(vtxIter->position().x(),vtxIter->position().y(),vtxIter->position().z());
      if (vertexPos.z()==theBeamSpot->position().z())
       { // in case vetex not found
        double sigmaZ=theBeamSpot->sigmaZ();
        double sigmaZ0Error=theBeamSpot->sigmaZ0Error();
        double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
        zmin1_=theBeamSpot->position().z()-nSigmasDeltaZ1_*sq;
        zmax1_=theBeamSpot->position().z()+nSigmasDeltaZ1_*sq;
       }
      else
       { // a vertex has been recoed
        zmin1_=vtxIter->position().z()-deltaZ1WithVertex_;
        zmax1_=vtxIter->position().z()+deltaZ1WithVertex_;
       }

      myMatchEle->set1stLayerZRange(zmin1_,zmax1_);
      myMatchPos->set1stLayerZRange(zmin1_,zmax1_);

      myMatchEle->setUseRecoVertex(true); //Hit matchers need to know that the vertex is known
      myMatchPos->setUseRecoVertex(true);

      // try electron
      double aCharge=-1.;

      if (!fromTrackerSeeds_)
       {
        std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits
         = myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);

        if (!elePixelHits.empty() )
         {
          LogDebug("ElectronSeedGenerator") << "seedsFromThisCluster: electron compatible hits found ";

          std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;

          for (v = elePixelHits.begin(); v != elePixelHits.end(); v++)
           {
            (*v).first.invert();
            bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,vertexPos);
            if (valid)
             {
              reco::ElectronSeed s(*pts_,recHits_,dir) ;
              s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
              result.push_back(s);
              delete pts_;
              pts_=0;
             }
           }
         }
       }
      else
       {
        std::vector<SeedWithInfo> elePixelSeeds=
        myMatchEle->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
        std::vector<SeedWithInfo>::iterator s;
        for (s = elePixelSeeds.begin(); s != elePixelSeeds.end(); s++)
         {
          reco::ElectronSeed seed(s->seed()) ;
          reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
          seed.setCaloCluster(caloCluster,s->subDet2(),s->dRz2(),s->dPhi2(),s->subDet1(),s->dRz1(),s->dPhi1(),hoe1,hoe2) ;
          result.push_back(seed);
	       }
       }

      // try positron
      aCharge=1.;

      if (!fromTrackerSeeds_)
       {
	      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits
	       = myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);

        if (!posPixelHits.empty() )
         {
          LogDebug("ElectronSeedGenerator") << "seedsFromThisCluster: positron compatible hits found ";

          std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
          for (v = posPixelHits.begin(); v != posPixelHits.end(); v++)
           {
            bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,vertexPos);
            if (valid)
             {
              reco::ElectronSeed s(*pts_,recHits_,dir) ;
              s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
              result.push_back(s);
              delete pts_;
              pts_=0;
             }
           }
         }
       }
      else
       {
        std::vector<SeedWithInfo> posPixelSeeds
         = myMatchPos->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos, clusterEnergy, aCharge);
        std::vector<SeedWithInfo>::iterator s;
        for (s = posPixelSeeds.begin(); s != posPixelSeeds.end(); s++)
         {
          reco::ElectronSeed seed(s->seed()) ;
          reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
          seed.setCaloCluster(caloCluster,s->subDet2(),s->dRz2(),s->dPhi2(),s->subDet1(),s->dRz1(),s->dPhi1(),hoe1,hoe2) ;
          result.push_back(seed);
         }
       }
     }
   }

  return ;
 }

bool ElectronSeedGenerator::prepareElTrackSeed
 ( ConstRecHitPointer innerhit,
   ConstRecHitPointer outerhit,
   const GlobalPoint& vertexPos )
 {

  // debug prints
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] outer PixelHit   x,y,z "<<outerhit->globalPosition();

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
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] final TSOS Pt: "<<updatedState_out.globalMomentum().perp();
  pts_ =  transformer_.persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
 }
