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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FastSimulation/EgammaElectronAlgos/interface/GSPixelHitMatcher.h" 
#include "FastSimulation/EgammaElectronAlgos/interface/ElectronGSPixelSeedGenerator.h" 

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"  
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include <vector>

ElectronGSPixelSeedGenerator::ElectronGSPixelSeedGenerator(
  const edm::ParameterSet &pset,
  double pTMin)
  :   
  dynamicphiroad_(pset.getParameter<bool>("dynamicPhiRoad")),
  lowPtThreshold_(pset.getParameter<double>("LowPtThreshold")),
  highPtThreshold_(pset.getParameter<double>("HighPtThreshold")),
  sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
  phimin2_(pset.getParameter<double>("PhiMin2")),      
  phimax2_(pset.getParameter<double>("PhiMax2")),
  deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
  deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
  deltaPhi2_(pset.getParameter<double>("DeltaPhi2")),
  pTMin2(pTMin*pTMin),
  myGSPixelMatcher(0),
  theUpdator(0), thePropagator(0), 
  //   theMeasurementTracker(0), 
  //   theNavigationSchool(0)
  theSetup(0), pts_(0)
{
  // Instantiate the pixel hit matcher
  myGSPixelMatcher = new GSPixelHitMatcher(pset.getParameter<double>("ePhiMin1"), 
					   pset.getParameter<double>("ePhiMax1"),
					   pset.getParameter<double>("pPhiMin1"),
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

}

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

  edm::ESHandle<TrackerInteractionGeometry> interGeom;
  setup.get<TrackerInteractionGeometryRecord>().get( interGeom );
  theTrackerInteractionGeometry = &(*interGeom);

  edm::ESHandle<MagneticFieldMap> fieldMap;
  setup.get<MagneticFieldMapRecord>().get(fieldMap);
  theMagneticFieldMap = &(*fieldMap);

  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 

  myGSPixelMatcher->setES(theMagneticFieldMap,
			  theTrackerGeometry,
			  theGeomSearchTracker,
			  theTrackerInteractionGeometry);

}

void  ElectronGSPixelSeedGenerator::run(
  edm::Event& e, 
  // const edm::Handle<reco::SuperClusterCollection>& clusters,  
  const reco::SuperClusterRefVector &sclRefs,
  const SiTrackerGSMatchedRecHit2DCollection* theGSRecHits,
  const edm::SimTrackContainer* theSimTracks,
  reco::ElectronPixelSeedCollection & out) {

  // Get the beam spot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByLabel("offlineBeamSpot",recoBeamSpotHandle); 

  // Get its position
  BSPosition_ = recoBeamSpotHandle->position();
  double sigmaZ=recoBeamSpotHandle->sigmaZ();
  double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
  double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  double zmin1 = BSPosition_.z()-3*sq;
  double zmax1 = BSPosition_.z()+3*sq;
  myGSPixelMatcher->set1stLayerZRange(zmin1,zmax1);

  // A map of vector of pixel seeds, for each clusters
  std::map<unsigned,std::vector<reco::ElectronPixelSeed> > myPixelSeeds;

  // No seeding attempted if no hits !
  if(theGSRecHits->size() == 0) return;    

  // The vector of simTrack Id's carrying GSRecHits
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

  // Loop over the simTrack carrying GSRecHits
  for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {

    unsigned simTrackId = theSimTrackIds[tkId];
    const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

    SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit2;

    // Request a minimum pT for the sim track
    if ( theSimTrack.momentum().perp2() < pTMin2 ) continue;

    // Request a minimum number of RecHits (total and in the pixel detector)
    unsigned numberOfRecHits = 0;
    
    // The vector of pixel rechis

    // Now save a collection of Pixel hits for seeding electrons
    std::vector<unsigned> layerHit(6,static_cast<unsigned>(0));
    const SiTrackerGSMatchedRecHit2D *hit;
    std::vector<ConstRecHitPointer> thePixelRecHits;
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	  iterRecHit != theRecHitRangeIteratorEnd; 
	  ++iterRecHit) { 
      ++numberOfRecHits;
      hit = &(*iterRecHit);
      const DetId& detId = iterRecHit->geographicalId();
      unsigned int theSubdetId = detId.subdetId(); 
      // Pixel hits only
      if( theSubdetId == PixelSubdetector::PixelBarrel || 
	  theSubdetId == PixelSubdetector::PixelEndcap ) { 

	// Check the layer hit (1-2-3 for barrel, 4-5 for forward)
	unsigned theHitLayer = 0;
	if ( theSubdetId ==  PixelSubdetector::PixelBarrel ) { 
	  PXBDetId pxbid(detId.rawId()); 
	  theHitLayer = pxbid.layer();  
	} else if ( theSubdetId ==  PixelSubdetector::PixelEndcap ) { 
	  PXFDetId pxfid(detId.rawId()); 
	  theHitLayer = pxfid.disk()+3;
	}

	// Keep only the first hit on a given layer (i.e., ignore overlaps)
	if ( !layerHit[theHitLayer] ) { 
	  layerHit[theHitLayer] = 1;
	  // Build the hit
	  const GeomDet* geomDet( theTrackerGeometry->idToDet(detId) );
	  ConstRecHitPointer aTrackingRecHit = 
	    GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit));
	  // Save the hit
	  thePixelRecHits.push_back(aTrackingRecHit);
	}
      }
    }    

    // At least 3 hits
    if ( numberOfRecHits < 3 ) continue;

    // At least 2 pixel hits
    if ( thePixelRecHits.size() < 2 ) continue;
    
    // Loop over clusters 

    unsigned csize = sclRefs.size();
    for  (unsigned int i=0;i<csize;++i) {
      
      // Find the pixel seeds (actually only the best one is returned)
      LogDebug ("run") << "new cluster, calling addAseedFromThisCluster";
      addASeedToThisCluster(sclRefs[i],thePixelRecHits,myPixelSeeds[i]);
      
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

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<sclRefs.size()
		   <<", no. of ElectronPixelSeeds found  = " << out.size();
  
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
  const GlobalPoint vertexPos(BSPosition_.x(),BSPosition_.y(),BSPosition_.z());

  LogDebug("") << "[ElectronGSPixelSeedGenerator::seedsFromThisCluster] " 
	       << "new supercluster with energy: " << clusterEnergy;
  LogDebug("") << "[ElectronGSPixelSeedGenerator::seedsFromThisCluster] " 
	       << "and position: " << clusterPos;

  //Here change the deltaPhi window of the first pixel layer in function of the seed pT
  if (dynamicphiroad_) {
    float clusterEnergyT = clusterEnergy*sin(seedCluster->position().theta()) ;
    
    float deltaPhi1 = 0.875/clusterEnergyT + 0.055; 
    if (clusterEnergyT < lowPtThreshold_) deltaPhi1= deltaPhi1Low_;
    if (clusterEnergyT > highPtThreshold_) deltaPhi1= deltaPhi1High_;
    
    float ephimin1 = -deltaPhi1*sizeWindowENeg_ ;
    float ephimax1 =  deltaPhi1*(1.-sizeWindowENeg_);
    float pphimin1 = -deltaPhi1*(1.-sizeWindowENeg_);
    float pphimax1 =  deltaPhi1*sizeWindowENeg_;
    
    float phimin2  = -deltaPhi2_/2. ;
    float phimax2  =  deltaPhi2_/2,;
    
    myGSPixelMatcher->set1stLayer(ephimin1,ephimax1,pphimin1,pphimax1);
    myGSPixelMatcher->set2ndLayer(phimin2,phimax2);
    
  }



  PropagationDirection dir = alongMomentum;
   
  // Find the best pixel pair compatible with the cluster
  std::vector<std::pair<ConstRecHitPointer,ConstRecHitPointer> > compatPixelHits = 
    myGSPixelMatcher->compatibleHits(clusterPos, vertexPos, clusterEnergy, thePixelRecHits);

  // The corresponding origin vertex
  float vertexZ = myGSPixelMatcher->getVertex();
  GlobalPoint theVertex(BSPosition_.x(),BSPosition_.y(),vertexZ); 
 
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
  // make a spiral from the two hits and the vertex position
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) return false;

  FreeTrajectoryState fts = helix.stateAtVertex();

  // Give infinite errors to start the fit (no pattern recognition here). 
  AlgebraicSymMatrix errorMatrix(5,1);
  fts.setCurvilinearError(errorMatrix*100.);

   TrajectoryStateOnSurface propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid()) return false;

  // The persitent trajectory state
  pts_ =  transformer_.persistentState(propagatedState, innerhit->geographicalId().rawId());

  // The corresponding rechits
  recHits_.clear();
  recHits_.push_back(innerhit->hit()->clone());
  recHits_.push_back(outerhit->hit()->clone());  


  return true;

}

