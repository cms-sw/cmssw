#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FastSimulation/Tracking/plugins/FastTrackMerger.h"

#include <vector>
#include <map>
//

//for debug only 
//#define FAMOS_DEBUG

FastTrackMerger::FastTrackMerger(const edm::ParameterSet& conf) 
{  
#ifdef FAMOS_DEBUG
  std::cout << "FastTrackMerger created" << std::endl;
#endif

  // The main product is a track collection, and all extras
  produces<reco::TrackCollection>();
  
  // The name of the track producers to merge
  trackProducers = conf.getParameter<std::vector<edm::InputTag> >("TrackProducers");

  // The name of the track producers to remove
  std::vector<edm::InputTag> defaultRemove;
  removeTrackProducers = 
    conf.getUntrackedParameter<std::vector<edm::InputTag> >("RemoveTrackProducers",defaultRemove);

  // Only the tracks!
  tracksOnly = conf.getUntrackedParameter<bool>("SaveTracksOnly",false);

  // optional pT cut
  double pTMin = conf.getUntrackedParameter<bool>("pTMin",0.);
  pTMin2 = pTMin*pTMin;

  // optional nHit cut
  minHits = conf.getUntrackedParameter<unsigned>("minHits",0);

  // optional track quality saving
  promoteQuality = conf.getUntrackedParameter<bool>("promoteTrackQuality",false);
  qualityStr = conf.getUntrackedParameter<std::string>("newQuality","");

  if ( !tracksOnly ) { 
    produces<reco::TrackExtraCollection>();
    produces<TrackingRecHitCollection>();
    produces<std::vector<Trajectory> >();
    produces<TrajTrackAssociationCollection>();
  }
}

  
// Functions that gets called by framework every event
void 
FastTrackMerger::produce(edm::Event& e, const edm::EventSetup& es) {        

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " FastTrackMerger produce init " << std::endl;
#endif

  // The produced objects
  std::auto_ptr<reco::TrackCollection> recoTracks(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> recoTrackExtras(new reco::TrackExtraCollection);
  std::auto_ptr<TrackingRecHitCollection> recoHits(new TrackingRecHitCollection);
  std::auto_ptr<std::vector<Trajectory> > recoTrajectories(new std::vector<Trajectory>);
  std::auto_ptr<TrajTrackAssociationCollection> recoTrajTrackMap(new TrajTrackAssociationCollection());

  // No seed -> output an empty track collection
  if(trackProducers.size() == 0) {
    e.put(recoTracks);
    if ( !tracksOnly ) { 
      e.put(recoTrackExtras);
      e.put(recoHits);
      e.put(recoTrajectories);
      e.put(recoTrajTrackMap);
    }
    return;
  }

  // The quality to be set
  reco::TrackBase::TrackQuality qualityToSet;
  if (qualityStr != "")
    qualityToSet = reco::TrackBase::qualityByName(qualityStr);
  else 
    qualityToSet = reco::TrackBase::undefQuality;

  // The input track collection handle
  edm::Handle<reco::TrackCollection> theTrackCollection;

  // First, the tracks to be removed
  std::set<unsigned> removeTracks;
  for ( unsigned aProducer=0; aProducer<removeTrackProducers.size(); ++aProducer ) { 
    bool isTrackCollection = e.getByLabel(removeTrackProducers[aProducer],theTrackCollection); 
    if (!isTrackCollection) continue;
    reco::TrackCollection::const_iterator aTrack = theTrackCollection->begin();
    reco::TrackCollection::const_iterator lastTrack = theTrackCollection->end();
    for ( ; aTrack!=lastTrack; ++aTrack ) {
      // Get the simtrack Id
      int recoTrackId = findId(*aTrack);
      if ( recoTrackId < 0 ) continue;
      // Remove the track if requested
      if ( removeTracks.find((unsigned)recoTrackId) != removeTracks.end() ) continue;
      removeTracks.insert((unsigned)recoTrackId);
    }      
  }
  
  // Then the tracks to be added
  std::set<unsigned> alreadyAddedTracks;
  
  // Loop on the track producers to be merged
  for ( unsigned aProducer=0; aProducer<trackProducers.size(); ++aProducer ) { 
    
    bool isTrackCollection = e.getByLabel(trackProducers[aProducer],theTrackCollection); 
    if ( ! isTrackCollection ) { 
#ifdef FAMOS_DEBUG
      std::cout << "***FastTrackMerger*** Warning! The track collection " 
		<< trackProducers[aProducer].encode() 
		<< " does not exist." << std::endl;
#endif
      continue;
    }

#ifdef FAMOS_DEBUG
    std::cout << "***FastTrackMerger*** of track collection " 
	      << trackProducers[aProducer].encode() 
	      << " with " << theTrackCollection->size() 
	      << " tracks to be copied"
	      << std::endl;
#endif
    reco::TrackCollection::const_iterator aTrack = theTrackCollection->begin();
    reco::TrackCollection::const_iterator lastTrack = theTrackCollection->end();

    // Only tracks are to be copied -> loop on the track collection and copy
    if ( tracksOnly ) { 

      // edm:: Handle<reco::TrackExtraCollection > theTrackExtraCollection;
      // bool isTrackExtraCollection = e.getByLabel(trackProducers[aProducer],theTrackExtraCollection); 
      bool index = 0;
      for ( ; aTrack!=lastTrack; ++aTrack,++index ) {

	// Find the track id
	int recoTrackId = findId(*aTrack);
	if ( recoTrackId < 0 ) continue;
      	
	// Ignore tracks to be removed or tracks already copied
	std::set<unsigned>::iterator iR = removeTracks.find((unsigned)recoTrackId);
#ifdef FAMOS_DEBUG
	if( iR != removeTracks.end() ) std::cout << recoTrackId << "(REMOVED) ";
#endif
	if( iR != removeTracks.end() ) continue;
	
	// Ignore tracks already copied
	std::set<unsigned>::iterator iA = alreadyAddedTracks.find((unsigned)recoTrackId);
#ifdef FAMOS_DEBUG
	if( iA != alreadyAddedTracks.end() ) std::cout << recoTrackId << "(ALREADY ADDED) ";
#endif
	if( iA != alreadyAddedTracks.end() ) continue;
	
#ifdef FAMOS_DEBUG
	std::cout << recoTrackId << " ";
#endif

	// Ignore tracks with too small a pT
	if ( aTrack->innerMomentum().Perp2() < pTMin2 ) continue;
	
	// Ignore tracks with too small a pT
	if ( aTrack->recHitsSize() < minHits ) continue;
	
	// A copy of the track + save the transient reference to the track extra reference
	reco::Track aRecoTrack(*aTrack);
	// const reco::TrackExtraRef theTrackExtraRef(*theTrackExtraCollection,index);
	// if ( isTrackExtraCollection ) aRecoTrack.setExtra(theTrackExtraRef);
	recoTracks->push_back(aRecoTrack);
	// Save the quality if requested
	if (promoteQuality) recoTracks->back().setQuality(qualityToSet);	
	
      }
      

    // All extras are to be copied too -> loop on the Trajectory/Track map association 
    } else { 

      edm:: Handle<std::vector<Trajectory> > theTrajectoryCollection;
      edm::Handle<TrajTrackAssociationCollection> theAssoMap;  

      // Count the number of hits and reserve appropriate memory
      unsigned nRecoHits = 0;
      for ( ; aTrack!=lastTrack; ++aTrack ) nRecoHits+= aTrack->recHitsSize();
      recoHits->reserve(nRecoHits); // This is to save some time at push_back.
      
      e.getByLabel(trackProducers[aProducer],theTrajectoryCollection);
      e.getByLabel(trackProducers[aProducer],theAssoMap);
      
      // The track collection iterators.
      TrajTrackAssociationCollection::const_iterator anAssociation;  
      TrajTrackAssociationCollection::const_iterator lastAssociation;
      anAssociation = theAssoMap->begin();
      lastAssociation = theAssoMap->end();
#ifdef FAMOS_DEBUG
      std::cout << "List of tracks to be copied " << std::endl;
#endif
      // Build the map of correspondance between reco tracks and sim tracks
      for ( ; anAssociation != lastAssociation; ++anAssociation ) { 
	edm::Ref<std::vector<Trajectory> > aTrajectoryRef = anAssociation->key;
	reco::TrackRef aTrackRef = anAssociation->val;
	// Find the track id
	int recoTrackId = findId(*aTrackRef);
	if ( recoTrackId < 0 ) continue;
      	
	// Ignore tracks to be removed or tracks already copied
	std::set<unsigned>::iterator iR = removeTracks.find((unsigned)recoTrackId);
#ifdef FAMOS_DEBUG
	if( iR != removeTracks.end() ) std::cout << recoTrackId << "(REMOVED) ";
#endif
	if( iR != removeTracks.end() ) continue;
	
	// Ignore tracks already copied
	std::set<unsigned>::iterator iA = alreadyAddedTracks.find((unsigned)recoTrackId);
#ifdef FAMOS_DEBUG
	if( iA != alreadyAddedTracks.end() ) std::cout << recoTrackId << "(ALREADY ADDED) ";
#endif
	if( iA != alreadyAddedTracks.end() ) continue;
	
#ifdef FAMOS_DEBUG
	std::cout << recoTrackId << " ";
#endif
	
	// Ignore tracks with too small a pT
	if ( aTrackRef->innerMomentum().Perp2() < pTMin2 ) continue;
	
	// Ignore tracks with too few hits
	if ( aTrackRef->recHitsSize() < minHits ) continue;
	
	// A copy of the track
	reco::Track aRecoTrack(*aTrackRef);
	recoTracks->push_back(aRecoTrack);      
	// Save the quality if requested
	if (promoteQuality) recoTracks->back().setQuality(qualityToSet);	
	// A copy of the hits
	unsigned nh = aRecoTrack.recHitsSize();
	for ( unsigned ih=0; ih<nh; ++ih ) {
	  TrackingRecHit *hit = aRecoTrack.recHit(ih)->clone();
	  recoHits->push_back(hit);
	}
	
	// A copy of the trajectories
	recoTrajectories->push_back(*aTrajectoryRef);
	
      }
#ifdef FAMOS_DEBUG
      std::cout << std::endl;
#endif
    }
}
    
    // Save the track candidates in the event
#ifdef FAMOS_DEBUG
  std::cout << "Saving " 
	    << recoTracks->size() << " reco::Tracks " << std::endl;
#endif
  
  if ( tracksOnly ) { 
    // Save only the tracks (with transient reference to track extras)
    e.put(recoTracks);

  } else { 
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

}

int 
FastTrackMerger::findId(const reco::Track& aTrack) const {
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


