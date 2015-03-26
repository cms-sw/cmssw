#include "TrackListCombiner.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

using namespace std;

/*****************************************************************************/
TrackListCombiner::TrackListCombiner(const edm::ParameterSet& ps)
{
  trackProducers = ps.getParameter<vector<string> >("trackProducers");

  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
  produces<vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
}

/*****************************************************************************/
TrackListCombiner::~TrackListCombiner()
{
}

/*****************************************************************************/
void TrackListCombiner::produce(edm::Event& ev, const edm::EventSetup& es)
{
  auto_ptr<reco::TrackCollection>          recoTracks
      (new reco::TrackCollection);
  auto_ptr<reco::TrackExtraCollection>     recoTrackExtras
      (new reco::TrackExtraCollection);
  auto_ptr<TrackingRecHitCollection>       recoHits
      (new TrackingRecHitCollection);
  auto_ptr<vector<Trajectory> >            recoTrajectories
      (new vector<Trajectory>);
  auto_ptr<TrajTrackAssociationCollection> recoTrajTrackMap
      (new TrajTrackAssociationCollection());

  LogTrace("MinBiasTracking")
    << "[TrackListCombiner]";

  // Go through all track producers
  int i = 1;
  for(vector<string>::iterator trackProducer = trackProducers.begin();
                               trackProducer!= trackProducers.end();
                               trackProducer++, i++)
  {
    reco::TrackBase::TrackAlgorithm algo;
    switch(i) 
    {
      case 1:  algo = reco::TrackBase::lowPtTripletStep; break;
      case 2:  algo = reco::TrackBase::pixelPairStep; break;
      case 3:  algo = reco::TrackBase::detachedTripletStep; break;
      default: algo = reco::TrackBase::undefAlgorithm;
    }

    edm::Handle<vector<Trajectory> > theTrajectoryCollection;
    edm::Handle<TrajTrackAssociationCollection> theAssoMap;  

    ev.getByLabel(*trackProducer, theTrajectoryCollection);
    ev.getByLabel(*trackProducer, theAssoMap);

    LogTrace("MinBiasTracking")
      << " [TrackListCombiner] " << *trackProducer
      << " : " << theAssoMap->size();

    
    // The track collection iterators
    TrajTrackAssociationCollection::const_iterator anAssociation;  
    TrajTrackAssociationCollection::const_iterator lastAssociation;
    anAssociation = theAssoMap->begin();
    lastAssociation = theAssoMap->end();

    // Build the map of correspondance between reco tracks and sim tracks
    for ( ; anAssociation != lastAssociation; ++anAssociation )
    { 
      edm::Ref<vector<Trajectory> > aTrajectoryRef = anAssociation->key;
      reco::TrackRef aTrackRef = anAssociation->val;
      
      // A copy of the track
      reco::Track aRecoTrack(*aTrackRef);

      // Set algorithm
      aRecoTrack.setAlgorithm(algo);

      recoTracks->push_back(aRecoTrack);      

      // A copy of the hits
      unsigned nh = aRecoTrack.recHitsSize();
      for(unsigned ih=0; ih<nh; ++ih)
      {
        TrackingRecHit *hit = aRecoTrack.recHit(ih)->clone();
        recoHits->push_back(hit);
      }
      
      // A copy of the trajectories
      recoTrajectories->push_back(*aTrajectoryRef);
      
    }
  }

  LogTrace("MinBiasTracking")
    << " [TrackListCombiner] allTracks : " << recoTracks->size()
                                    << "|" << recoTrajectories->size();

  // Save the tracking recHits
  edm::OrphanHandle<TrackingRecHitCollection> theRecoHits = ev.put(recoHits);
  
  edm::RefProd<TrackingRecHitCollection> theRecoHitsProd(theRecoHits);
  // Create the track extras and add the references to the rechits
  unsigned hits = 0;
  unsigned nTracks = recoTracks->size();
  recoTrackExtras->reserve(nTracks); // To save some time at push_back
  for(unsigned index = 0; index < nTracks; ++index )
  { 
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
    aTrackExtra.setHits(theRecoHitsProd,hits,nHits);
    hits +=nHits;

    recoTrackExtras->push_back(aTrackExtra);
  }
  
  // Save the track extras
  edm::OrphanHandle<reco::TrackExtraCollection> theRecoTrackExtras =
    ev.put(recoTrackExtras);
  
  // Add the reference to the track extra in the tracks
  for(unsigned index = 0; index<nTracks; ++index)
  { 
    const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
    (recoTracks->at(index)).setExtra(theTrackExtraRef);
  }
  
  // Save the tracks
  edm::OrphanHandle<reco::TrackCollection> theRecoTracks = ev.put(recoTracks);
  
  // Save the trajectories
  edm::OrphanHandle<vector<Trajectory> > theRecoTrajectories =
    ev.put(recoTrajectories);
  
  // Create and set the trajectory/track association map 
  for(unsigned index = 0; index<nTracks; ++index)
  { 
    edm::Ref<vector<Trajectory> > trajRef( theRecoTrajectories, index );
    edm::Ref<reco::TrackCollection>    tkRef( theRecoTracks, index );
    recoTrajTrackMap->insert(trajRef,tkRef);
  }
  
  // Save the association map
  ev.put(recoTrajTrackMap);
}

