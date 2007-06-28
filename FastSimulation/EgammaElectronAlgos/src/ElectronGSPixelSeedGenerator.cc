// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronGSPixelSeedGenerator.
// 
/**\class ElectronGSPixelSeedGenerator EgammaElectronAlgos/ElectronGSPixelSeedGenerator

 Description: Top algorithm producing ElectronPixelSeeds, ported from FAMOS

 Implementation:
     future redesign...
*/
//
// Original Author:  Patrick Janot
//
//
#include "FastSimulation/EgammaElectronAlgos/interface/ElectronGSPixelSeedGenerator.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FastSimulation/EgammaElectronAlgos/interface/GSPixelHitMatcher.h" 
#include "FastSimulation/EgammaElectronAlgos/interface/ElectronGSPixelSeedGenerator.h" 

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"  
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include <vector>

ElectronGSPixelSeedGenerator::ElectronGSPixelSeedGenerator(
  float iephimin1, float iephimax1,
  float ipphimin1, float ipphimax1,
  float iphimin2, float iphimax2,
  float izmin1, float izmax1,
  float izmin2, float izmax2,
  bool idynamicphiroad)
  : 
  ephimin1(iephimin1), ephimax1(iephimax1), 
  pphimin1(ipphimin1), pphimax1(ipphimax1), 
  phimin2(iphimin2), phimax2(iphimax2),
  zmin1(izmin1), zmax1(izmax1),
  zmin2(izmin2), zmax2(izmax2),
  dynamicphiroad(idynamicphiroad),
  myGSPixelMatcher(0),
  theMode_(unknown), 
  theUpdator(0), thePropagator(0), 
  //   theMeasurementTracker(0), 
  //   theNavigationSchool(0)
  theSetup(0), pts_(0)
{}

ElectronGSPixelSeedGenerator::~ElectronGSPixelSeedGenerator() {

  //  delete theNavigationSchool;
  delete myGSPixelMatcher;
  //  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;

}


void ElectronGSPixelSeedGenerator::setupES(const edm::EventSetup& setup) {

  theSetup= &setup;

  edm::ESHandle<MagneticField> pMF;
  setup.get<IdealMagneticFieldRecord>().get(pMF);
  theMagField = &(*pMF);

  edm::ESHandle<TrackerGeometry>        geometry;
  setup.get<TrackerDigiGeometryRecord>().get(geometry);
  theTrackerGeometry = &(*geometry);

  edm::ESHandle<GeometricSearchTracker> recoGeom;
  setup.get<TrackerRecoGeometryRecord>().get( recoGeom );
  theGeomSearchTracker = &(*recoGeom);

  theUpdator = new KFUpdator();
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 
  //  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));

  //  edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
  //  setup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  //  theMeasurementTracker = measurementTrackerHandle.product();

  //  myMatchEle->setES(&(*theMagField),theMeasurementTracker);
  //  myMatchPos->setES(&(*theMagField),theMeasurementTracker);

  myGSPixelMatcher->setES(theMagField,theTrackerGeometry,theGeomSearchTracker);

}

void  ElectronGSPixelSeedGenerator::run(
  edm::Event& e, 
  const edm::Handle<reco::SuperClusterCollection> &clusters, 
  reco::ElectronPixelSeedCollection & out) {

  // A map of vector of pixel seeds, for each clusters
  std::map<unsigned,std::vector<reco::ElectronPixelSeed> > myPixelSeeds;

  // Get the Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSTC;
  e.getByLabel("famosSimHits",theSTC);
  const edm::SimTrackContainer* theSimTracks = &(*theSTC);
  
  // Get the Monte Carlo truth (SimVertices)
  //edm::Handle<edm::SimVertexContainer> theSVC;
  //e.getByLabel("famosSimHits",theSVC);
  //SimVertexContainer* theSimVertices = &(*theSTC);

  // Get the collection of Tracker RecHits
  edm::Handle<SiTrackerGSRecHit2DCollection> theRHC;
  e.getByLabel("siTrackerGaussianSmearingRecHits", theRHC);
  const SiTrackerGSRecHit2DCollection* theGSRecHits = &(*theRHC);

  // No seeding attempted if no hits !
  if(theGSRecHits->size() == 0) return;    

  // The vector of simTrack Id's carrying GSRecHits
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

  // Loop over the simTrack carrying GSRecHits
  for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {

    unsigned simTrackId = theSimTrackIds[tkId];
    const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

    SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit2;

    // Request a minimum pT for the sim track
    if ( theSimTrack.momentum().perp2() < 1.0 ) continue;

    // Request a minimum number of RecHits (total and in the pixel detector)
    unsigned numberOfRecHits = 0;
    
    // The vector of pixel rechis

    // Now save a collection of Pixel hits for seeding electrons
    const SiTrackerGSRecHit2D *hit;
    std::vector<ConstRecHitPointer> thePixelRecHits;
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	  iterRecHit != theRecHitRangeIteratorEnd; 
	  ++iterRecHit) { 
      ++numberOfRecHits;
      hit = &(*iterRecHit);
      // Pixel hits only
      if( (unsigned int)(hit->geographicalId().subdetId())== PixelSubdetector::PixelBarrel || 
	  (unsigned int)(hit->geographicalId().subdetId())== PixelSubdetector::PixelEndcap ) { 
	const DetId& detId =  iterRecHit->geographicalId();
	const GeomDet* geomDet( theTrackerGeometry->idToDet(detId) );
	  //	  const TrackingRecHit* a TrackingRecHit = 
	  //	  GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit))->hit()->clone();
	ConstRecHitPointer aTrackingRecHit = 
	  GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit));

	thePixelRecHits.push_back(aTrackingRecHit);
      }
    }    

    // At least 3 hits
    if ( numberOfRecHits < 3 ) continue;

    // At least 2 pixel hits
    if ( thePixelRecHits.size() < 2 ) continue;
    
    // Loop over clusters 

    unsigned csize = clusters->size();
    for  (unsigned int i=0;i<csize;++i) {
      
      edm::Ref<reco::SuperClusterCollection> theClusB(clusters,i);
      
      // Find the pixel seeds (actually only the best one is returned)
      LogDebug ("run") << "new cluster, calling addAseedFromThisCluster";
      addASeedToThisCluster(theClusB,thePixelRecHits,myPixelSeeds[i]);
      
    }

  }

  // Back to the expected collection
 
  std::map<unsigned,std::vector<reco::ElectronPixelSeed> >::const_iterator px = myPixelSeeds.begin();
  std::map<unsigned,std::vector<reco::ElectronPixelSeed> >::const_iterator pxEnd = myPixelSeeds.end();
  for ( ; px!=pxEnd; ++px ) {
    unsigned nSeeds = (px->second).size();
    for ( unsigned ipx = 0; ipx<nSeeds; ++ipx ) { 
      out.push_back((px->second)[ipx]); 
      reco::ElectronPixelSeed is = px->second[ipx];
    }
  }

  if(theMode_==offline) LogDebug ("run") << "(offline)";
  
  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<clusters->size()
   <<", no. of ElectronPixelSeeds found  = " << out.size();
  
}

void ElectronGSPixelSeedGenerator::setup(bool off)
{

  if(theMode_==unknown)
    {
      // Instantiate the pixel hit matcher
      LogDebug("") << "ElectronGSPixelSeedGenerator, phi limits: " 
		   << ephimin1 << ", " << ephimax1 << ", "
		   << pphimin1 << ", " << pphimax1;
      myGSPixelMatcher = new GSPixelHitMatcher( 
			       ephimin1, ephimax1, 
			       pphimin1, pphimax1,
			       phimin2,  phimax2,
			       zmin1, zmax1, zmin2, zmax2);
      //      myMatchPos = new PixelHitMatcher( pphimin1, pphimax1, pphimin2, pphimax2, 
      //					zmin1, zmax1, zmin2, zmax2);
      if(off) theMode_=offline; else theMode_ = HLT;
    }

}

void ElectronGSPixelSeedGenerator::addASeedToThisCluster( 
  edm::Ref<reco::SuperClusterCollection> seedCluster, 
  std::vector<ConstRecHitPointer>& thePixelRecHits,
  std::vector<reco::ElectronPixelSeed>& result)
{
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(),
			 seedCluster->position().y(), 
			 seedCluster->position().z());
  const GlobalPoint vertexPos(0.,0.,0.);

  LogDebug("") << "[ElectronGSPixelSeedGenerator::seedsFromThisCluster] " 
	       << "new supercluster with energy: " << clusterEnergy;
  LogDebug("") << "[ElectronGSPixelSeedGenerator::seedsFromThisCluster] " 
	       << "and position: " << clusterPos;

  //Here change the deltaPhi window of the first pixel layer in function of the seed pT
  if (dynamicphiroad)
    {
      float clusterEnergyT = clusterEnergy*sin(seedCluster->position().theta()) ;

      float deltaPhi1 = 1.4/clusterEnergyT ;
      float deltaPhi2 = 0.7/clusterEnergyT ;
      float ephimin1 = -deltaPhi1*0.625 ;
      float ephimax1 =  deltaPhi1*0.375 ;
      float pphimin1 = -deltaPhi1*0.375 ;
      float pphimax1 =  deltaPhi1*0.625 ;
      float phimin2  = -deltaPhi2*0.5 ;
      float phimax2  =  deltaPhi2*0.5 ;

      if (clusterEnergyT < 5) {

	ephimin1 = -0.280*0.625 ;
	ephimax1 =  0.280*0.375 ;
	pphimin1 = -0.280*0.375 ;
	pphimax1 =  0.280*0.625 ;
	phimin2  = -0.007 ;
	phimin2  =  0.007 ;

      } else if (clusterEnergyT > 35) {
	
	ephimin1 = -0.040*0.625 ;
	ephimax1 =  0.040*0.375 ;
	pphimin1 = -0.040*0.375 ;
	pphimax1 =  0.040*0.625 ;
	phimin2  = -0.001 ;
	phimax2  =  0.001 ;
	
      }

      //      myMatchEle->set1stLayer(ephimin1,ephimax1);
      //      myMatchPos->set1stLayer(pphimin1,pphimax1);
      //      myMatchEle->set2ndLayer(phimin2,phimax2);
      //      myMatchPos->set2ndLayer(phimin2,phimax2);

      myGSPixelMatcher->set1stLayer(ephimin1,ephimax1,pphimin1,pphimax1);
      myGSPixelMatcher->set2ndLayer(phimin2,phimax2);

    }


  PropagationDirection dir = alongMomentum;
   
  // Find the best pixel pair compatible with the cluster
  std::vector<std::pair<ConstRecHitPointer,ConstRecHitPointer> > compatPixelHits = 
    myGSPixelMatcher->compatibleHits(clusterPos, vertexPos, clusterEnergy, thePixelRecHits);

  // The corresponding origin vertex
  float vertexZ = myGSPixelMatcher->getVertex();
  GlobalPoint theVertex(0.,0.,vertexZ); 
 
  // Create the Electron pixel seed.
  if (!compatPixelHits.empty() ) {
    LogDebug("") << "[ElectronGSPixelSeedGenerator::seedsFromThisCluster] " 
		 << " electron compatible hits found ";
    std::vector<std::pair<ConstRecHitPointer,ConstRecHitPointer> >::iterator v;
    for (v = compatPixelHits.begin(); v != compatPixelHits.end(); ++v ) {
       
      bool valid = prepareElTrackSeed(v->first,v->second, theVertex);
      if (valid) {
        reco::ElectronPixelSeed s= reco::ElectronPixelSeed(seedCluster,*pts_,recHits_,dir);
        result.push_back(s);
	delete pts_;
	pts_=0;
      }
    }
  }  
  
  // And return !
  return ;

}

bool ElectronGSPixelSeedGenerator::prepareElTrackSeed(ConstRecHitPointer innerhit,
						      ConstRecHitPointer outerhit,
						      const GlobalPoint& vertexPos)
{
  
  // debug prints
  LogDebug("") <<"[ElectronGSPixelSeedGenerator::prepareElTrackSeed] " 
	       << "inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[ElectronGSPixelSeedGenerator::prepareElTrackSeed] " 
	       << "outer PixelHit   x,y,z "<<outerhit->globalPosition();

  pts_=0;
  recHits_.clear();
    
  recHits_.push_back(innerhit->hit()->clone());
  recHits_.push_back(outerhit->hit()->clone());  

  typedef TrajectoryStateOnSurface     TSOS;
  // make a spiral
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) return false;

  FreeTrajectoryState fts = helix.stateAtVertex();
  TSOS propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid()) return false;

  TSOS updatedState = theUpdator->update(propagatedState, *innerhit);
  
  TSOS propagatedState_out = thePropagator->propagate(fts,outerhit->det()->surface()) ;
  if (!propagatedState_out.isValid()) return false;

  TSOS updatedState_out = theUpdator->update(propagatedState_out, *outerhit);

  // debug prints
  LogDebug("") <<"[ElectronGSPixelSeedGenerator::prepareElTrackSeed] " 
	       << "final TSOS, position: "<< updatedState_out.globalPosition() 
	       << " momentum: " << updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronGSPixelSeedGenerator::prepareElTrackSeed] " 
	       << "final TSOS Pt: "<<updatedState_out.globalMomentum().perp();

  pts_ =  transformer_.persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
}
