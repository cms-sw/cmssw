#include <memory>
#include <string>

#include "FastSimulation/EgammaElectronAlgos/plugins/GSGsfTrackCandidateMaker.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateSeedAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

GSGsfTrackCandidateMaker::GSGsfTrackCandidateMaker(const edm::ParameterSet& conf)
{  
  // What is supposed to be produced
  produces<TrackCandidateCollection>();  
  produces<reco::TrackCandidateSeedAssociationCollection>();

  // The pixel seeds origin
  seedProducer = conf.getParameter<std::string>("SeedProducer");
  seedLabel = conf.getParameter<std::string>("SeedLabel");

  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("overlapCleaning");

  // Kinematic cuts
  minimumNumberOfHits = conf.getParameter<int>("minimumNumberOfHits");
  ptCut = conf.getParameter<double>("ptCut");

}


// Virtual destructor needed.
GSGsfTrackCandidateMaker::~GSGsfTrackCandidateMaker() {}  

void 
GSGsfTrackCandidateMaker::beginJob (const edm::EventSetup& es)
{
  //services
  edm::ESHandle<TrackerGeometry>        geometry;
  es.get<TrackerDigiGeometryRecord>().get(geometry);
  theTrackerGeometry = &(*geometry);

}

// Functions that gets called by framework every event
void 
GSGsfTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es)
{        
  
  // Step A: Retrieve inputs
  // 1) Pixel Seeds
  edm::Handle<TrajectorySeedCollection> seedCollection;
  e.getByLabel(seedProducer, seedLabel, seedCollection);
  const TrajectorySeedCollection* theSeedCollection = &(*seedCollection);
  
  // 2) Tracker RecHits
  edm::Handle<SiTrackerGSRecHit2DCollection> theRHC;
  e.getByLabel("siTrackerGaussianSmearingRecHits", theRHC);
  const SiTrackerGSRecHit2DCollection* theGSRecHits = &(*theRHC);

  // 3) The Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSTC;
  e.getByLabel("famosSimHits",theSTC);
  const edm::SimTrackContainer* theSimTracks = &(*theSTC);
  
  // Step B: Create empty output collections
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
  std::auto_ptr<reco::TrackCandidateSeedAssociationCollection> outAssoc(new reco::TrackCandidateSeedAssociationCollection);    

  // Step C: Loop over the pixel seeds. Keep only the first one for each track
  //         (Equivalent to seed cleaning in the case of Gsf!)
  //         Create one track candidate for each of these seeds.
  std::vector<int> seedLocations;
  int theCurrentTrackId = -1;
  int seednr = -1;
  int seedkept = 0;

  if ( theSeedCollection->size() > 0 ) { 

    TrajectorySeedCollection::const_iterator iseed = theSeedCollection->begin();
    TrajectorySeedCollection::const_iterator theLastSeed = theSeedCollection->end();
    
    for( ; iseed != theLastSeed; ++iseed ) {
      ++seednr;
      // The first hit of the seed  and its simtrack id
      BasicTrajectorySeed::const_iterator theFirstHit = iseed->recHits().first;
      BasicTrajectorySeed::const_iterator theSecondHit = theFirstHit; ++theSecondHit;
      const SiTrackerGSRecHit2D* rechit1 = (const SiTrackerGSRecHit2D*) &(*theFirstHit) ;
      int theSimTrackId = rechit1->simtrackId();

      // The simulated Track
      const SimTrack& theSimTrack = (*theSimTracks)[theSimTrackId];
      double ptSim = theSimTrack.momentum().Pt();
      //      const SiTrackerGSRecHit2D* rechit2 = (const SiTrackerGSRecHit2D*) &(*theSecondHit) ;
      // The DetId's for later comparison
      const DetId& detId1 =  theFirstHit->geographicalId();
      const DetId& detId2 =  theSecondHit->geographicalId();

      // Don't consider seeds belonging to a track already considered 
      // (Equivalent to seed cleaning)
      if ( theCurrentTrackId == theSimTrackId ) continue;
      theCurrentTrackId = theSimTrackId;

      // The RecHits corresponding to that track
      SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(theSimTrackId);
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit2;

      // Create OwnVector with sorted GSRecHit's
      unsigned int previousLayerNumber = 0;
      unsigned int previousSubdetId = 0;
      double previousLambda1 = 0.;
      bool replacePreviousHit = false;
      bool keepThisHit = true;
      TrackingRecHit* previousHit = 0;
      //
      edm::OwnVector<TrackingRecHit> recHits;
      //
      for ( iterRecHit = theRecHitRangeIteratorBegin; 
	    iterRecHit != theRecHitRangeIteratorEnd; 
	    ++iterRecHit) {
	
	// The DetId
	const DetId& detId =  iterRecHit->geographicalId();

	// Decide between two hits on the same layer 
	if ( rejectOverlaps ) { 
	  
	  unsigned int subdetId = detId.subdetId(); 
	  unsigned int layerNumber=0;
	  if ( subdetId == StripSubdetector::TIB) { 
	    TIBDetId tibid(detId.rawId()); 
	    layerNumber = tibid.layer();
	  } else if ( subdetId ==  StripSubdetector::TOB ) { 
	    TOBDetId tobid(detId.rawId()); 
	    layerNumber = tobid.layer();
	  } else if ( subdetId ==  StripSubdetector::TID) { 
	    TIDDetId tidid(detId.rawId());
	    layerNumber = tidid.wheel();
	  } else if ( subdetId ==  StripSubdetector::TEC ) { 
	    TECDetId tecid(detId.rawId()); 
	    layerNumber = tecid.wheel(); 
	  } else if ( subdetId ==  PixelSubdetector::PixelBarrel ) { 
	    PXBDetId pxbid(detId.rawId()); 
	    layerNumber = pxbid.layer();  
	  } else if ( subdetId ==  PixelSubdetector::PixelEndcap ) { 
	    PXFDetId pxfid(detId.rawId()); 
	    layerNumber = pxfid.disk();  
	  }
	  
	  // Determine the smaller uncertainty for the current hit
	  double xx = iterRecHit->localPositionError().xx();
	  double yy = iterRecHit->localPositionError().yy();
	  double xy = iterRecHit->localPositionError().xy();
	  double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
	  double lambda1 = 0.5 * (xx+yy-delta);
	  
	  // Not the same layer : Add the current hit
	  if ( subdetId != previousSubdetId || layerNumber != previousLayerNumber ) {
	    keepThisHit = true;
	    replacePreviousHit = false;
	    // Same layer : keep the better hit, and drop the other 
	  } else {
	    // In that case, the current hit is better -> drop the previous one
	    if ( lambda1 < previousLambda1 ) {
	      keepThisHit = true;
	      replacePreviousHit = true;
	      // In this case, the previous hit was better -> do nothing ! (ignore the current hit)
	    } else {
	      keepThisHit = false;
	      replacePreviousHit = false;
	    }
	  }
	  previousSubdetId = subdetId;
	  previousLayerNumber = layerNumber;
	  previousLambda1 = lambda1;
	}
	
	// Don't add early pixel hits if not from the seed
	unsigned savedHits = recHits.size();
	if ( savedHits == 0 && detId==detId1 || 
	     savedHits == 1 && detId==detId2 || 
	     savedHits > 1 ) {

	  // delete previous hit
	  if ( replacePreviousHit ) {
	    delete previousHit;
	    recHits.pop_back();
	  }
	  
	  // add current hit
	  if ( keepThisHit ) { 
	    
	    const GeomDet* geomDet( theTrackerGeometry->idToDet(detId) );
	    TrackingRecHit* aTrackingRecHit = 
	      GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit))->hit()->clone();
	    recHits.push_back(aTrackingRecHit);
	    previousHit = aTrackingRecHit;

	    // ignore current hit
	  } else { 
	    previousHit = 0;
	  }

	}
      }

      // Add the track candidate if it has enough hits and large enough a pT
      if ( recHits.size() >= minimumNumberOfHits && ptSim > ptCut ) { 

	output->push_back(TrackCandidate(recHits,*iseed,iseed->startingState()));
	seedLocations.push_back(seednr);

	// Count the seed selected (and the track candidates)
	++seedkept;

      }

    }
      
  }
    
  // Step D: write TrackCandidateCollection to the event and create Associationmap
  const edm::OrphanHandle<TrackCandidateCollection> refprodTrackC = e.put(output);

  for (unsigned int i=0;i<seedLocations.size();++i) {
    outAssoc->insert(edm::Ref<TrackCandidateCollection>(refprodTrackC,i),
		     edm::Ref<TrajectorySeedCollection>(seedCollection,seedLocations[i]));    
  }
  
  // Step E: write AssociationMap to the event
  e.put(outAssoc);

  // Step F: There we go out of this mess!
  return;

}


