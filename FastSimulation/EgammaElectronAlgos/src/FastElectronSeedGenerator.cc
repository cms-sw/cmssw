// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      FastElectronSeedGenerator.
//
/**\class FastElectronSeedGenerator EgammaElectronAlgos/FastElectronSeedGenerator

 Description: Top algorithm producing ElectronSeeds, ported from FAMOS

 Implementation:
     future redesign...
*/
//
// Original Author:  Patrick Janot
//
//
#include "FastElectronSeedGenerator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FastPixelHitMatcher.h"
#include "FastElectronSeedGenerator.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include <vector>

//#define FAMOS_DEBUG

FastElectronSeedGenerator::FastElectronSeedGenerator(
  const edm::ParameterSet &pset,
  double pTMin,
  const edm::InputTag& beamSpot)
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
  fromTrackerSeeds_(pset.getParameter<bool>("fromTrackerSeeds")),
  theUpdator(0), thePropagator(0),
  //   theMeasurementTracker(0),
  //   theNavigationSchool(0)
  theSetup(0), theBeamSpot(beamSpot)
{

#ifdef FAMOS_DEBUG
  std::cout << "FromTrackerSeeds  = " << fromTrackerSeeds_ << std::endl;
#endif

  // Instantiate the pixel hit matcher
  searchInTIDTEC = pset.getParameter<bool>("searchInTIDTEC");
  myGSPixelMatcher = new FastPixelHitMatcher(pset.getParameter<double>("ePhiMin1"),
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

FastElectronSeedGenerator::~FastElectronSeedGenerator() {

  //  delete theNavigationSchool;
  delete myGSPixelMatcher;
  //  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;

}


void FastElectronSeedGenerator::setupES(const edm::EventSetup& setup) {

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

  thePropagator = new PropagatorWithMaterial(alongMomentum,.000511,&(*theMagField));

  myGSPixelMatcher->setES(theMagneticFieldMap,
			  theTrackerGeometry,
			  theGeomSearchTracker,
			  theTrackerInteractionGeometry);

}

void  FastElectronSeedGenerator::run(edm::Event& e,
				      const reco::SuperClusterRefVector &sclRefs,
				      const SiTrackerGSMatchedRecHit2DCollection* theGSRecHits,
				      const edm::SimTrackContainer* theSimTracks,
				      TrajectorySeedCollection *seeds,
				      reco::ElectronSeedCollection & out){

  // Take the seed collection.
  theInitialSeedColl=seeds;

  // Get the beam spot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByLabel(theBeamSpot,recoBeamSpotHandle);

  // Get its position
  BSPosition_ = recoBeamSpotHandle->position();
  double sigmaZ=recoBeamSpotHandle->sigmaZ();
  double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
  double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  double zmin1 = BSPosition_.z()-3*sq;
  double zmax1 = BSPosition_.z()+3*sq;
#ifdef FAMOS_DEBUG
  std::cout << "Z Range for pixel matcher : " << zmin1 << " " << BSPosition_.z() << " " << zmax1 << std::endl;
#endif
  myGSPixelMatcher->set1stLayerZRange(zmin1,zmax1);

  // A map of vector of pixel seeds, for each clusters
  std::map<unsigned,std::vector<reco::ElectronSeed> > myPixelSeeds;

  // No seeding attempted if no hits !
  if(theGSRecHits->size() == 0) return;

  if ( !fromTrackerSeeds_ ) {

    // The vector of simTrack Id's carrying GSRecHits
    const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

    // Loop over the simTrack carrying GSRecHits
    for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {

      unsigned simTrackId = theSimTrackIds[tkId];
      const SimTrack& theSimTrack = (*theSimTracks)[simTrackId];

      // Request a minimum pT for the sim track
      if ( theSimTrack.momentum().perp2() < pTMin2 ) continue;

      // Request a minimum number of RecHits (total and in the pixel detector)
      unsigned numberOfRecHits = 0;

      // The vector of rechits for seeding

      // 1) Cluster-pixel match seeding:
      //    Save a collection of Pixel +TEC +TID hits for seeding electrons
      std::vector<unsigned> layerHit(6,static_cast<unsigned>(0));
      // const SiTrackerGSMatchedRecHit2D *hit;
      TrackerRecHit currentHit;
      std::vector<TrackerRecHit> theHits;
      TrajectorySeed theTrackerSeed;

      SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
      SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
      SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit2;

      for ( iterRecHit = theRecHitRangeIteratorBegin;
	    iterRecHit != theRecHitRangeIteratorEnd;
	    ++iterRecHit) {
	++numberOfRecHits;

	currentHit = TrackerRecHit(&(*iterRecHit),theTrackerGeometry);
	if ( ( currentHit.subDetId() <= 2 ) ||  // Pixel Hits
	     // Add TID/TEC (optional)
	     ( searchInTIDTEC &&
	       ( ( currentHit.subDetId() == 3 &&
		   currentHit.ringNumber() < 3 &&
		   currentHit.layerNumber() < 3 ) || // TID first two rings, first two layers
		 ( currentHit.subDetId() == 6 &&
		   currentHit.ringNumber() < 3 &&
		   currentHit.layerNumber() < 3 ) ) ) ) // TEC first two rings, first two layers
	  theHits.push_back(currentHit);
      }

      // At least 3 hits
      if ( numberOfRecHits < 3 ) continue;

      // At least 2 pixel hits
      if ( theHits.size() < 2 ) continue;

      // Loop over clusters

      unsigned csize = sclRefs.size();
      for  (unsigned int i=0;i<csize;++i) {

	// Find the pixel seeds (actually only the best one is returned)
	LogDebug ("run") << "new cluster, calling addAseedFromThisCluster";
	addASeedToThisCluster(sclRefs[i],theHits,theTrackerSeed,myPixelSeeds[i]);

      }

    }
  // 2) Check if the seed is in the a-priori seed collection
  } else {

    // Loop over the tracker seed
#ifdef FAMOS_DEBUG
    std::cout << "We have " << seeds->size() << " tracker seeds!" << std::endl;
#endif
    for (unsigned int i=0;i<seeds->size();++i) {

      TrackerRecHit currentHit;
      std::vector<TrackerRecHit> theHits;
      const TrajectorySeed& theTrackerSeed = (*seeds)[i];
      TrajectorySeed::range theSeedRange=theTrackerSeed.recHits();
      TrajectorySeed::const_iterator theSeedRangeIteratorBegin = theSeedRange.first;
      TrajectorySeed::const_iterator theSeedRangeIteratorEnd   = theSeedRange.second;
      TrajectorySeed::const_iterator theSeedItr = theSeedRangeIteratorBegin;

      for ( ; theSeedItr != theSeedRangeIteratorEnd; ++theSeedItr ) {
	const SiTrackerGSMatchedRecHit2D * theSeedingRecHit =
	  (const SiTrackerGSMatchedRecHit2D*) (&(*theSeedItr));
	currentHit = TrackerRecHit(theSeedingRecHit,theTrackerGeometry);
	theHits.push_back(currentHit);
      }

      // Loop over clusters
      unsigned csize = sclRefs.size();
      for  (unsigned int i=0;i<csize;++i) {

	// Find the pixel seeds (actually only the best one is returned)
#ifdef FAMOS_DEBUG
	std::cout << "new cluster, calling addAseedFromThisCluster" << std::endl;
#endif
	addASeedToThisCluster(sclRefs[i],theHits,theTrackerSeed,myPixelSeeds[i]);

      }
      // End loop over clusters
    }
    // End loop over seeds
  }
  // end else

  // Back to the expected collection

  std::map<unsigned,std::vector<reco::ElectronSeed> >::const_iterator px = myPixelSeeds.begin();
  std::map<unsigned,std::vector<reco::ElectronSeed> >::const_iterator pxEnd = myPixelSeeds.end();
  for ( ; px!=pxEnd; ++px ) {
    unsigned nSeeds = (px->second).size();
    for ( unsigned ipx = 0; ipx<nSeeds; ++ipx ) {
      out.push_back((px->second)[ipx]);
      reco::ElectronSeed is = px->second[ipx];
    }
  }

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<sclRefs.size()
		   <<", no. of ElectronSeeds found  = " << out.size();
#ifdef FAMOS_DEBUG
  std::cout << ": For event "<<e.id() << std::endl;
  std::cout <<"Nr of superclusters: "<<sclRefs.size()
	    <<", no. of ElectronSeeds found  = " << out.size() << std::endl;
#endif

}

void
FastElectronSeedGenerator::addASeedToThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
						    std::vector<TrackerRecHit>& theHits,
						    const TrajectorySeed& theTrackerSeed,
						    std::vector<reco::ElectronSeed>& result)
{

  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(),
			 seedCluster->position().y(),
			 seedCluster->position().z());
  const GlobalPoint vertexPos(BSPosition_.x(),BSPosition_.y(),BSPosition_.z());

#ifdef FAMOS_DEBUG
  std::cout << "[FastElectronSeedGenerator::seedsFromThisCluster] "
	    << "new supercluster with energy: " << clusterEnergy << std::endl;
  std::cout << "[FastElectronSeedGenerator::seedsFromThisCluster] "
	    << "and position: " << clusterPos << std::endl;
  std::cout << "Vertex position : " << vertexPos << std::endl;
#endif

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

    float phimin2  = -deltaPhi2_/2.;
    float phimax2  =  deltaPhi2_/2.;

    myGSPixelMatcher->set1stLayer(ephimin1,ephimax1,pphimin1,pphimax1);
    myGSPixelMatcher->set2ndLayer(phimin2,phimax2);

  }



  PropagationDirection dir = alongMomentum;

  // Find the best pixel pair compatible with the cluster
  std::vector<std::pair<ConstRecHitPointer,ConstRecHitPointer> > compatPixelHits =
    myGSPixelMatcher->compatibleHits(clusterPos, vertexPos, clusterEnergy, theHits);

  // The corresponding origin vertex
  double vertexZ = myGSPixelMatcher->getVertex();
  GlobalPoint theVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);

  // Create the Electron pixel seed.
  if (!compatPixelHits.empty() ) {
#ifdef FAMOS_DEBUG
    std::cout << "[FastElectronSeedGenerator::seedsFromThisCluster] "
	      << " electron compatible hits found " << std::endl;
#endif
    // Pixel-matching case: create the seed from scratch
    if (!fromTrackerSeeds_) {

      std::vector<std::pair<ConstRecHitPointer,ConstRecHitPointer> >::iterator v;
      for (v = compatPixelHits.begin(); v != compatPixelHits.end(); ++v ) {

		bool valid = prepareElTrackSeed(v->first,v->second, theVertex);
		if (valid) {
		  reco::ElectronSeed s(pts_,recHits_,dir) ;
		  s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
		  result.push_back(s);
	    }

      }

    // Here we take instead the seed from a-priori seed collection
    } else {

      reco::ElectronSeed s(theTrackerSeed);
	  s.setCaloCluster(reco::ElectronSeed::CaloClusterRef(seedCluster)) ;
	  result.push_back(s);
    }

  }

#ifdef FAMOS_DEBUG
    else
      std::cout << "[FastElectronSeedGenerator::seedsFromThisCluster] "
      		<< " No electron compatible hits found " << std::endl;
#endif


  // And return !
  return ;

}

bool FastElectronSeedGenerator::prepareElTrackSeed(ConstRecHitPointer innerhit,
						      ConstRecHitPointer outerhit,
						      const GlobalPoint& vertexPos)
{

  // debug prints
  LogDebug("") <<"[FastElectronSeedGenerator::prepareElTrackSeed] "
	       << "inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[FastElectronSeedGenerator::prepareElTrackSeed] "
	       << "outer PixelHit   x,y,z "<<outerhit->globalPosition();

  // make a spiral from the two hits and the vertex position
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) return false;

  FreeTrajectoryState fts = helix.stateAtVertex();

  // Give infinite errors to start the fit (no pattern recognition here).
  AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();
  fts.setCurvilinearError(errorMatrix*100.);

   TrajectoryStateOnSurface propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid()) return false;

  // The persitent trajectory state
  pts_ =  trajectoryStateTransform::persistentState(propagatedState, innerhit->geographicalId().rawId());

  // The corresponding rechits
  recHits_.clear();
  recHits_.push_back(innerhit->hit()->clone());
  recHits_.push_back(outerhit->hit()->clone());


  return true;

}

