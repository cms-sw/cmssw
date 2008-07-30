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
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
//#include "FastSimulation/Tracking/interface/TrackerRecHitSplit.h"

#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"

#include <vector>
#include <map>
//

//for debug only 
//#define FAMOS_DEBUG

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf) 
{  
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer created" << std::endl;
#endif

  // The main product is a track candidate collection.
  produces<TrackCandidateCollection>();

  // These products contain tracks already reconstructed at this level
  // (No need to reconstruct them twice!)
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
  
  // The name of the seed producer
  seedProducer = conf.getParameter<edm::InputTag>("SeedProducer");

  // The name of the recHit producer
  hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

  // The name of the track producer (tracks already produced need not be produced again!)
  // trackProducer = conf.getParameter<edm::InputTag>("TrackProducer");
  trackProducers = conf.getParameter<std::vector<edm::InputTag> >("TrackProducers");

  // Copy (or not) the tracks already produced in a new collection
  keepFittedTracks = conf.getParameter<bool>("KeepFittedTracks");

  // The minimum number of crossed layers
  minNumberOfCrossedLayers = conf.getParameter<unsigned int>("MinNumberOfCrossedLayers");

  // The maximum number of crossed layers
  maxNumberOfCrossedLayers = conf.getParameter<unsigned int>("MaxNumberOfCrossedLayers");

  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("OverlapCleaning");

  // Split hits ?
  splitHits = conf.getParameter<bool>("SplitHits");

  // Reject tracks with several seeds ?
  // Typically don't do that at HLT for electrons, but do it otherwise
  seedCleaning = conf.getParameter<bool>("SeedCleaning");

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

  // Useful typedef's to avoid retyping
  typedef std::pair<reco::TrackRef,edm::Ref<std::vector<Trajectory> > > TrackPair;
  typedef std::map<unsigned,TrackPair> TrackMap;

  // The produced objects
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
  std::auto_ptr<reco::TrackCollection> recoTracks(new reco::TrackCollection);    
  std::auto_ptr<TrackingRecHitCollection> recoHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> recoTrackExtras(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> > recoTrajectories(new std::vector<Trajectory> );
  std::auto_ptr<TrajTrackAssociationCollection> recoTrajTrackMap( new TrajTrackAssociationCollection() );
  
  // Get the seeds
  // edm::Handle<TrajectorySeedCollection> theSeeds;
  edm::Handle<edm::View<TrajectorySeed> > theSeeds;
  e.getByLabel(seedProducer,theSeeds);

  // No seed -> output an empty track collection
  if(theSeeds->size() == 0) {
    e.put(output);
    e.put(recoTracks);
    e.put(recoHits);
    e.put(recoTrackExtras);
    e.put(recoTrajectories);
    e.put(recoTrajTrackMap);
    return;
  }

  // Get the GS RecHits
  //  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
  e.getByLabel(hitProducer, theGSRecHits);

  // The input track collection + extra's
  /*
  edm::Handle<reco::TrackCollection> theTrackCollection;
  edm:: Handle<std::vector<Trajectory> > theTrajectoryCollection;
  edm::Handle<TrajTrackAssociationCollection> theAssoMap;  
  bool isTrackCollection = e.getByLabel(trackProducer,theTrackCollection);
  */
  std::vector<edm::Handle<reco::TrackCollection> > theTrackCollections;
  std::vector<edm:: Handle<std::vector<Trajectory> > > theTrajectoryCollections;
  std::vector<edm::Handle<TrajTrackAssociationCollection> > theAssoMaps;
  std::vector<bool> isTrackCollections;
  TrajTrackAssociationCollection::const_iterator anAssociation;  
  TrajTrackAssociationCollection::const_iterator lastAssociation;
  TrackMap theTrackMap;
  unsigned nCollections = trackProducers.size();
  unsigned nRecoHits = 0;

  if ( nCollections ) { 
    theTrackCollections.resize(nCollections);
    theTrajectoryCollections.resize(nCollections);
    theAssoMaps.resize(nCollections);
    isTrackCollections.resize(nCollections);
    for ( unsigned tprod=0; tprod < nCollections; ++tprod ) { 
      isTrackCollections[tprod] = e.getByLabel(trackProducers[tprod],theTrackCollections[tprod]); 

      if ( isTrackCollections[tprod] ) { 
	// The track collection
	reco::TrackCollection::const_iterator aTrack = theTrackCollections[tprod]->begin();
	reco::TrackCollection::const_iterator lastTrack = theTrackCollections[tprod]->end();
	// The numbers of hits
	for ( ; aTrack!=lastTrack; ++aTrack ) nRecoHits+= aTrack->recHitsSize();
	e.getByLabel(trackProducers[tprod],theTrajectoryCollections[tprod]);
	e.getByLabel(trackProducers[tprod],theAssoMaps[tprod]);
	// The association between trajectories and tracks
	anAssociation = theAssoMaps[tprod]->begin();
	lastAssociation = theAssoMaps[tprod]->end(); 
#ifdef FAMOS_DEBUG
	std::cout << "Input Track Producer : " << trackProducer << std::endl;
	std::cout << "List of tracks already reconstructed " << std::endl;
#endif
	// Build the map of correspondance between reco tracks and sim tracks
	for ( ; anAssociation != lastAssociation; ++anAssociation ) { 
	  edm::Ref<std::vector<Trajectory> > aTrajectoryRef = anAssociation->key;
	  reco::TrackRef aTrackRef = anAssociation->val;
	  // Find the simtrack id of the reconstructed track
	  int recoTrackId = findId(*aTrackRef);
	  if ( recoTrackId < 0 ) continue;
#ifdef FAMOS_DEBUG
	  std::cout << recoTrackId << " ";
#endif
	  // And store it.
	  theTrackMap[recoTrackId] = TrackPair(aTrackRef,aTrajectoryRef);
	}
#ifdef FAMOS_DEBUG
	std::cout << std::endl;
#endif
      }
    }
    // This is to save some time at push_back.
    recoHits->reserve(nRecoHits); 
  }

  // Loop over the seeds
  int currentTrackId = -1;
  /*
  TrajectorySeedCollection::const_iterator aSeed = theSeeds->begin();
  TrajectorySeedCollection::const_iterator lastSeed = theSeeds->end();
  for ( ; aSeed!=lastSeed; ++aSeed ) { 
    // The first hit of the seed  and its simtrack id
  */
  /* */
#ifdef FAMOS_DEBUG
  std::cout << "Input seed Producer : " << seedProducer << std::endl;
#endif
  unsigned seed_size = theSeeds->size(); 
  for (unsigned seednr = 0; seednr < seed_size; ++seednr){

    // The seed
    const BasicTrajectorySeed* aSeed = &((*theSeeds)[seednr]);
  /* */
    // Find the first hit of the Seed
    TrajectorySeed::range theSeedingRecHitRange = aSeed->recHits();
    //    const SiTrackerGSRecHit2D * theFirstSeedingRecHit = 
    //      (const SiTrackerGSRecHit2D*) (&(*(theSeedingRecHitRange.first)));
    const SiTrackerGSMatchedRecHit2D * theFirstSeedingRecHit = 
      (const SiTrackerGSMatchedRecHit2D*) (&(*(theSeedingRecHitRange.first)));

    TrackerRecHit theFirstSeedingTrackerRecHit(theFirstSeedingRecHit,theGeometry);
    // SiTrackerGSRecHit2DCollection::const_iterator theSeedingRecHitEnd = theSeedingRecHitRange.second;

    // The SimTrack id associated to that recHit
    int simTrackId = theFirstSeedingRecHit->simtrackId();
    // std::cout << "The Sim Track Id : " << simTrackId << std::endl;
    // std::cout << "The Current Track Id : " << currentTrackId << std::endl;
    // const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

    // Don't consider seeds belonging to a track already considered 
    // (Equivalent to seed cleaning)
    if ( seedCleaning && currentTrackId == simTrackId ) continue;
    currentTrackId = simTrackId;
    
    // A vector of TrackerRecHits belonging to the track and the number of crossed layers
    std::vector<TrackerRecHit> theTrackerRecHits;
    // std::vector<TrackerRecHit> theTrackerRecHitsSplit;
    unsigned theNumberOfCrossedLayers = 0;
 
    // The track has indeed been reconstructed already -> Save the pertaining info
    TrackMap::const_iterator theTrackIt = theTrackMap.find(simTrackId);
    //    if ( isTrackCollection && theTrackIt != theTrackMap.end() ) { 
    if ( nCollections && theTrackIt != theTrackMap.end() ) { 

      if ( keepFittedTracks ) { 

#ifdef FAMOS_DEBUG
	std::cout << "Track " << simTrackId << " already reconstructed -> copy it" << std::endl;
#endif      
	// The track and trajectroy references
	reco::TrackRef aTrackRef = theTrackIt->second.first;
	edm::Ref<std::vector<Trajectory> > aTrajectoryRef = theTrackIt->second.second;
	
	// A copy of the track
	reco::Track aRecoTrack(*aTrackRef);
	recoTracks->push_back(aRecoTrack);      
	
	// A copy of the hits
	unsigned nh = aRecoTrack.recHitsSize();
	for ( unsigned ih=0; ih<nh; ++ih ) {
	  TrackingRecHit *hit = aRecoTrack.recHit(ih)->clone();
	  recoHits->push_back(hit);
	}
	
	// A copy of the trajectories
	recoTrajectories->push_back(*aTrajectoryRef);
	
      } else { 

#ifdef FAMOS_DEBUG
	std::cout << "Track " << simTrackId << " already reconstructed -> ignore it" << std::endl;
#endif      

      }

      // The track was not saved -> create a track candidate.

    } else { 
      
#ifdef FAMOS_DEBUG
      std::cout << "Track " << simTrackId << " will return a track candidate" << std::endl;
#endif
      // Get all the rechits associated to this track
      SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
      SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
      
      bool firstRecHit = true;
      // 
      TrackerRecHit theCurrentRecHit, thePreviousRecHit;

      TrackerRecHit theFirstHitComp, theSecondHitComp;

      for ( iterRecHit = theRecHitRangeIteratorBegin; 
	    iterRecHit != theRecHitRangeIteratorEnd; 
	    ++iterRecHit) {
	
	// Check the number of crossed layers
	if ( theNumberOfCrossedLayers >= maxNumberOfCrossedLayers ) continue;
	
	// Get current and previous rechits
	thePreviousRecHit = theCurrentRecHit;
	theCurrentRecHit = TrackerRecHit(&(*iterRecHit),theGeometry);
	
	// Check that the first rechit is indeed the first seeding hit
	if ( firstRecHit && theCurrentRecHit != theFirstSeedingTrackerRecHit ) continue;
	
	// Count the number of crossed layers
	if ( !theCurrentRecHit.isOnTheSameLayer(thePreviousRecHit) ) 
	  ++theNumberOfCrossedLayers;
	
	// Add all rechits (Grouped Trajectory Builder) from this hit onwards
	// Always add the first seeding rechit anyway
	if ( !rejectOverlaps || firstRecHit ) {  
	  
	  // Split matched hits (if requested / possible )
	  if ( splitHits && theCurrentRecHit.matchedHit()->isMatched() ) {
	    
	    addSplitHits(theCurrentRecHit,theTrackerRecHits);
	    
	  // No splitting   
	  } else {
	    
	    theTrackerRecHits.push_back(theCurrentRecHit);

	  }

	  firstRecHit = false;
	  
	// And now treat the following RecHits if hits in the same layer 
	// have to be rejected - The split option is not 
	} else { 
	  
	  // Not the same layer : Add the current hit
	  if ( theCurrentRecHit.subDetId()    != thePreviousRecHit.subDetId() || 
	       theCurrentRecHit.layerNumber() != thePreviousRecHit.layerNumber() ) {
	    
	    // Split matched hits (if requested / possible )
	    if ( splitHits && theCurrentRecHit.matchedHit()->isMatched() ) {
	      
	      addSplitHits(theCurrentRecHit,theTrackerRecHits);

	    // No splitting   	      
	    } else {

	      theTrackerRecHits.push_back(theCurrentRecHit);
	    
	    }
	    
	    // Same layer : keep the current hit if better, and drop the other - otherwise do nothing  
	  } else if ( theCurrentRecHit.localError() < thePreviousRecHit.localError() ) { 
	    
	    // Split matched hits (if requested / possible )
	    if( splitHits && theCurrentRecHit.matchedHit()->isMatched() ){

	      // Remove the previous hit(s)
	      theTrackerRecHits.pop_back();
	      if ( thePreviousRecHit.matchedHit()->isMatched() ) theTrackerRecHits.pop_back();

	      // Replace by the new hits
	      addSplitHits(theCurrentRecHit,theTrackerRecHits);

	    // No splitting   
	    } else {

	      // Replace the previous hit by the current hit
	      theTrackerRecHits.back() = theCurrentRecHit;

	    }

	  } else { 
	    
	  }

	}
	
      }
      // End of loop over the track rechits
    }
    
#ifdef FAMOS_DEBUG
     std::cout << "Hit number " << theTrackerRecHits.size() << std::endl;
#endif

    //replace TrackerRecHit con TrackerRecHitsSplit
    // 1) Create the OwnWector of TrackingRecHits
    edm::OwnVector<TrackingRecHit> recHits;
    unsigned nTrackerHits = theTrackerRecHits.size();
    recHits.reserve(nTrackerHits); // To save some time at push_back

    for ( unsigned ih=0; ih<nTrackerHits; ++ih ) {
      TrackingRecHit* aTrackingRecHit = theTrackerRecHits[ih].hit()->clone();
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
      if ( theTrackerRecHits[ih].matchedHit() && theTrackerRecHits[ih].matchedHit()->isMatched() ) 
	std::cout << "Matched : " << theTrackerRecHits[ih].matchedHit()->isMatched() 
		  << "Rphi Hit = " <<  theTrackerRecHits[ih].matchedHit()->monoHit()->simhitId()		 
		  << "Stereo Hit = " <<  theTrackerRecHits[ih].matchedHit()->stereoHit()->simhitId()
		  <<std::endl;

#endif
    }

    // Check the number of crossed layers
    if ( theNumberOfCrossedLayers < minNumberOfCrossedLayers ) continue;


    // Create a track Candidate (now with the reference to the seed!) .
    TrackCandidate  
      newTrackCandidate(recHits, 
			*aSeed, 
			aSeed->startingState(), 
			edm::RefToBase<TrajectorySeed>(theSeeds,seednr));

    //std::cout << "Track kept for later fit!" << std::endl;
    
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
#ifdef FAMOS_DEBUG
  std::cout << "Saving " 
	    << output->size() << " track candidates and " 
	    << recoTracks->size() << " reco::Tracks " << std::endl;
#endif
  // Save the track candidates
  e.put(output);



  // Save the tracking recHits

  edm::OrphanHandle <TrackingRecHitCollection> theRecoHits = e.put(recoHits );

  // Create the track extras and add the references to the rechits
  unsigned hits=0;
  unsigned nTracks = recoTracks->size();
  recoTrackExtras->reserve(nTracks); // To save some time at push_back
  for ( unsigned index = 0; index < nTracks; ++index ) { 
    //reco::TrackExtra aTrackExtra;
    reco::Track& aTrack = recoTracks->at(index);
    reco::TrackExtra aTrackExtra(aTrack.outerPosition(),
				 aTrack.outerMomentum(),
				 aTrack.outerOk(),
				 aTrack.innerPosition(),
				 aTrack.innerMomentum(),
				 aTrack.innerOk(),
				 aTrack.outerStateCovariance(),
				 aTrack.outerDetId(),
				 aTrack.innerStateCovariance(),
				 aTrack.innerDetId(),
				 aTrack.seedDirection(),
				 aTrack.seedRef());

    unsigned nHits = aTrack.recHitsSize();
    for ( unsigned int ih=0; ih<nHits; ++ih) {
      aTrackExtra.add(TrackingRecHitRef(theRecoHits,hits++));
    }
    recoTrackExtras->push_back(aTrackExtra);
  }
  

  // Save the track extras
  edm::OrphanHandle<reco::TrackExtraCollection> theRecoTrackExtras = e.put(recoTrackExtras);

  // Add the reference to the track extra in the tracks
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
    (recoTracks->at(index)).setExtra(theTrackExtraRef);
  }

  // Save the tracks
  edm::OrphanHandle<reco::TrackCollection> theRecoTracks = e.put(recoTracks);

  // Save the trajectories
  edm::OrphanHandle<std::vector<Trajectory> > theRecoTrajectories = e.put(recoTrajectories);
  
  // Create and set the trajectory/track association map 
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    edm::Ref<std::vector<Trajectory> > trajRef( theRecoTrajectories, index );
    edm::Ref<reco::TrackCollection>    tkRef( theRecoTracks, index );
    recoTrajTrackMap->insert(trajRef,tkRef);
  }

  // Save the association map.
  e.put(recoTrajTrackMap);

}

int 
TrackCandidateProducer::findId(const reco::Track& aTrack) const {
  int trackId = -1;
  trackingRecHit_iterator aHit = aTrack.recHitsBegin();
  trackingRecHit_iterator lastHit = aTrack.recHitsEnd();
  for ( ; aHit!=lastHit; ++aHit ) {
    if ( !aHit->get()->isValid() ) continue;
    //    const SiTrackerGSRecHit2D * rechit = (const SiTrackerGSRecHit2D*) (aHit->get());
    const SiTrackerGSMatchedRecHit2D * rechit = (const SiTrackerGSMatchedRecHit2D*) (aHit->get());
    trackId = rechit->simtrackId();
    break;
  }
  return trackId;
}

void 
TrackCandidateProducer::addSplitHits(const TrackerRecHit& theCurrentRecHit,
				     std::vector<TrackerRecHit>& theTrackerRecHits) { 
  
  const SiTrackerGSRecHit2D* mHit = theCurrentRecHit.matchedHit()->monoHit();
  const SiTrackerGSRecHit2D* sHit = theCurrentRecHit.matchedHit()->stereoHit();
  
  // Add the new hits
  if( mHit->simhitId() < sHit->simhitId() ) {
    
    theTrackerRecHits.push_back(TrackerRecHit(mHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrackerRecHit(sHit,theCurrentRecHit));
    
  } else {
    
    theTrackerRecHits.push_back(TrackerRecHit(sHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrackerRecHit(mHit,theCurrentRecHit));
    
  }

}
