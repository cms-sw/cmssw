#include "RecoTracker/IterativeTracking/interface/QualityFilter.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// Class Declarations
//

using namespace edm;
using namespace reco;
using namespace std;

QualityFilter::QualityFilter(const ParameterSet& iConfig)
{
  copyExtras_ = iConfig.getUntrackedParameter<bool>("copyExtras", false);

  produces<TrackCollection>();
  if (copyExtras_) {
      produces<TrackingRecHitCollection>();
      produces<TrackExtraCollection>();
  }
  produces<vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  trajTag = consumes<vector<Trajectory> >(iConfig.getParameter<InputTag>("recTracks"));
  tassTag = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<InputTag>("recTracks"));
  trackQuality_= TrackBase::qualityByName(iConfig.getParameter<string>("TrackQuality"));
}

QualityFilter::~QualityFilter()
{
 
   // Do anything here that needs to be done at destruction time.
   // (e.g. close files, deallocate resources etc.)

}

//
// Member Functions
//

// ------------ Method called to produce the data  ------------
void
QualityFilter::produce(Event& iEvent, const EventSetup& iSetup)
{
  auto_ptr<TrackCollection> selTracks(new TrackCollection);
  auto_ptr<TrackingRecHitCollection> selHits(copyExtras_ ? new TrackingRecHitCollection() : 0);
  auto_ptr<TrackExtraCollection> selTrackExtras(copyExtras_ ? new TrackExtraCollection() : 0);
  auto_ptr<vector<Trajectory> > outputTJ(new vector<Trajectory> );
  auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
  
  TrackExtraRefProd rTrackExtras; 
  TrackingRecHitRefProd rHits;
  if (copyExtras_) {
    rTrackExtras = iEvent.getRefBeforePut<TrackExtraCollection>();
    rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
  }

  Handle<vector<Trajectory> > TrajectoryCollection;
  Handle<TrajTrackAssociationCollection> assoMap;
  
  iEvent.getByToken(trajTag,TrajectoryCollection);
  iEvent.getByToken(tassTag,assoMap);

  TrajTrackAssociationCollection::const_iterator it = assoMap->begin();
  TrajTrackAssociationCollection::const_iterator lastAssoc = assoMap->end();
  for( ; it != lastAssoc; ++it ) {
    const Ref<vector<Trajectory> > traj = it->key;
    const TrackRef itc = it->val;
    bool goodTk = (itc->quality(trackQuality_));
 
    if (goodTk){
      Track track =(*itc);
      //Tracks and Trajectories
      selTracks->push_back( track );
      outputTJ->push_back( *traj );
      if (copyExtras_) {
          //Tracking Hits
          trackingRecHit_iterator irhit   =(*itc).recHitsBegin();
          trackingRecHit_iterator lasthit =(*itc).recHitsEnd();
          for (; irhit!=lasthit; ++irhit) {
            selHits->push_back((*irhit)->clone() );
          }
      }
    }
  }

  unsigned nTracks = selTracks->size();
  if (copyExtras_) {
      //Put Tracking Hits In The Event.
      OrphanHandle<TrackingRecHitCollection> theRecoHits = iEvent.put(selHits );
  
      //Put Track Extra In The Event.
      selTrackExtras->reserve(nTracks);
      unsigned hits=0;

      for ( unsigned index = 0; index<nTracks; ++index ) { 

        reco::Track& aTrack = selTracks->at(index);
        TrackExtra aTrackExtra(aTrack.outerPosition(),
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
            
        //unsigned nHits = aTrack.numberOfValidHits();
        unsigned nHits = aTrack.recHitsSize();
        for ( unsigned int ih=0; ih<nHits; ++ih) {
          aTrackExtra.add(TrackingRecHitRef(theRecoHits,hits++));
        }
        selTrackExtras->push_back(aTrackExtra);
      }

      //Correct Ref to Track.
      OrphanHandle<TrackExtraCollection> theRecoTrackExtras = iEvent.put(selTrackExtras); 
      for ( unsigned index = 0; index<nTracks; ++index ) { 
        const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
        (selTracks->at(index)).setExtra(theTrackExtraRef);
      }
  } // End If Copy Extras.

  //Tracks and Trajectories
  OrphanHandle<TrackCollection> theRecoTracks = iEvent.put(selTracks);
  OrphanHandle<vector<Trajectory> > theRecoTrajectories = iEvent.put(outputTJ);

  //TRACKS<->TRAJECTORIES MAP 
  nTracks = theRecoTracks->size();
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    Ref<vector<Trajectory> > trajRef( theRecoTrajectories, index );
    Ref<TrackCollection>    tkRef( theRecoTracks, index );
    trajTrackMap->insert(trajRef,tkRef);
  }
  //Map In The Event.
  iEvent.put( trajTrackMap );
}

// ------------ Method called once each job, just after ending the event loop.  ------------
void 
QualityFilter::endJob() {
}
