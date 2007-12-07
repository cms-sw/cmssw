#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"
//

//for debug only 
//#define FAMOS_DEBUG

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf) 
{  
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer created" << std::endl;
#endif
  produces<TrackCandidateCollection>();
  
  // The name of the seed producer
  seedProducer = conf.getParameter<edm::InputTag>("SeedProducer");

  // The name of the recHit producer
  hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("overlapCleaning");

}

  
// Virtual destructor needed.
TrackCandidateProducer::~TrackCandidateProducer() {

  // do nothing
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer destructed" << std::endl;
#endif

} 
 
void 
TrackCandidateProducer::beginJob (edm::EventSetup const & es) {

  //services
  //  es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);

  edm::ESHandle<TrackerGeometry>        geometry;


  es.get<TrackerDigiGeometryRecord>().get(geometry);

  theGeometry = &(*geometry);

}
  
  // Functions that gets called by framework every event
void 
TrackCandidateProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " TrackCandidateProducer produce init " << std::endl;
#endif

  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
  
  edm::Handle<TrajectorySeedCollection> theSeeds;
  e.getByLabel(seedProducer,theSeeds);

  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  e.getByLabel(hitProducer, theGSRecHits);
  
  // No seed -> output an empty track collection
  if(theSeeds->size() == 0) {
    e.put(output);
    return;
  }
  
  // Loop over the seeds
  TrajectorySeedCollection::const_iterator aSeed = theSeeds->begin();
  TrajectorySeedCollection::const_iterator lastSeed = theSeeds->end();
  for ( ; aSeed!=lastSeed; ++aSeed ) { 

    // Find the first hit of the Seed
    TrajectorySeed::range theSeedingRecHitRange = aSeed->recHits();
    const SiTrackerGSRecHit2D * theFirstSeedingRecHit = 
      dynamic_cast<const SiTrackerGSRecHit2D *> (&(*(theSeedingRecHitRange.first)));
    TrackerRecHit theFirstSeedingTrackerRecHit(theFirstSeedingRecHit,theGeometry);
    // SiTrackerGSRecHit2DCollection::const_iterator theSeedingRecHitEnd = theSeedingRecHitRange.second;

    // The SimTrack id associated to that recHit
    unsigned simTrackId = theFirstSeedingRecHit->simtrackId();
    // const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

    // Get all the rechits associated to this track
    SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
    std::vector<TrackerRecHit> theTrackerRecHits;

    bool firstRecHit = true;
    // 
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	  iterRecHit != theRecHitRangeIteratorEnd; 
	  ++iterRecHit) {

      TrackerRecHit theCurrentRecHit(&(*iterRecHit),theGeometry);
      // Check that the first rechit is indeed the first seeding hit
      if ( firstRecHit && theCurrentRecHit != theFirstSeedingTrackerRecHit ) continue;

      // Add all rechits (Grouped Trajectory Builder) from this hit onwards
      // Always add the first seeding rechit anyway
      if ( !rejectOverlaps || firstRecHit ) {  
	
	theTrackerRecHits.push_back(theCurrentRecHit);
	firstRecHit = false;
	
      // And now treat the following RecHits if hits in the same layer 
      // have to be rejected
      } else { 

	// Not the same layer : Add the current hit
	if ( theCurrentRecHit.subDetId()    != theTrackerRecHits.back().subDetId() || 
	     theCurrentRecHit.layerNumber() != theTrackerRecHits.back().layerNumber() ) {
	  theTrackerRecHits.push_back(theCurrentRecHit);
	// Same layer : keep the current hit if better, and drop the other - otherwise do nothing  
	} else if ( theCurrentRecHit.localError() < theTrackerRecHits.back().localError() ) { 
	    theTrackerRecHits.back() = theCurrentRecHit;
#ifdef FAMOS_DEBUG
	    std::cout << "Hit number " << theTrackerRecHits.size() 
		      << " : The local error is smaller than the previous hit " 
		      << theCurrentRecHit.localError() << " " 
		      <<  theTrackerRecHits.back().localError() << " in subdet/layer/ring " 
		      << theCurrentRecHit.subDetId() << " " 
		      << theCurrentRecHit.layerNumber() << " " 
		      << theCurrentRecHit.ringNumber() << " -> REPLACE " << std::endl;
#endif
	} else { 
#ifdef FAMOS_DEBUG
	    std::cout << "Hit number " << theTrackerRecHits.size() 
		      << " : The local error is larger than the previous hit " 
		      << theCurrentRecHit.localError() << " " 
		      <<  theTrackerRecHits.back().localError() << " in subdet/layer/ring " 
		      << theCurrentRecHit.subDetId() << " " 
		      << theCurrentRecHit.layerNumber() << " " 
		      << theCurrentRecHit.ringNumber() << " -> IGNORE " << std::endl;
#endif
	}
      }
    // End of loop over the track rechits
    }

    // 1) Create the OwnWector of TrackingRecHits
    edm::OwnVector<TrackingRecHit> recHits;
    for ( unsigned ih=0; ih<theTrackerRecHits.size(); ++ih ) {
      TrackingRecHit* aTrackingRecHit = 
	  GenericTransientTrackingRecHit::build(theTrackerRecHits[ih].geomDet(),
						theTrackerRecHits[ih].hit())->hit()->clone();
      recHits.push_back(aTrackingRecHit);
#ifdef FAMOS_DEBUG
      const DetId& detId = theTrackerRecHits[ih].hit()->geographicalId();      
      std::cout << "Added RecHit from detid " << detId.rawId() 
		<< " subdet = " << theTrackerRecHits[ih].subDetId() 
		<< " layer = " << theTrackerRecHits[ih].layerNumber()
		<< " ring = " << theTrackerRecHits[ih].ringNumber()
		<< " error = " << theTrackerRecHits[ih].localError()
		<< std::endl;
      
      std::cout << "Track/z/r : "
		<< simTrackId << " " 
		<< theTrackerRecHits[ih].globalPosition().z() << " " 
		<< theTrackerRecHits[ih].globalPosition().perp() << std::endl;
#endif
    }

    // Create a track Candidate .
    TrackCandidate newTrackCandidate(recHits, *aSeed, aSeed->startingState());
    
#ifdef FAMOS_DEBUG
    // Log
    std::cout << "\tSeed Information " << std::endl;
    std::cout << "\tSeed Direction = " << aSeed->direction() << std::endl;
    std::cout << "\tSeed StartingDet = " << aSeed->startingState().detId() << std::endl;
    
    std::cout << "\tTrajectory Parameters " 
	      << std::endl;
    std::cout << "\t\t detId  = " 
	      << newTrackCandidate.trajectoryStateOnDet().detId() 
	      << std::endl;
    std::cout << "\t\t loc.px = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().x()    
	      << std::endl;
    std::cout << "\t\t loc.py = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().y()    
	      << std::endl;
    std::cout << "\t\t loc.pz = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().z()    
	      << std::endl;
    std::cout << "\t\t error  = ";
    for(std::vector< float >::const_iterator iElement = newTrackCandidate.trajectoryStateOnDet().errorMatrix().begin();
	iElement < newTrackCandidate.trajectoryStateOnDet().errorMatrix().end();
	++iElement) {
      std::cout << "\t" << *iElement;
    }
    std::cout << std::endl;
#endif

    output->push_back(newTrackCandidate);

  }
  
  // Save the track candidates in the event
  e.put(output);

}


