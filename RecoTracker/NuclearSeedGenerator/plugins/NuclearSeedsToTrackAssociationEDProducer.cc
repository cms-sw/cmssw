#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsToTrackAssociationEDProducer.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


NuclearSeedsToTrackAssociationEDProducer::NuclearSeedsToTrackAssociationEDProducer(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  produces<TrackToSeedsMap>();
    //produces<TrackToTracksMap>();
}

NuclearSeedsToTrackAssociationEDProducer::~NuclearSeedsToTrackAssociationEDProducer()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
NuclearSeedsToTrackAssociationEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   /// Get the primary tracks
   edm::Handle<reco::TrackCollection>  primaryTrackCollection;
   iEvent.getByLabel( "TrackRefitter", primaryTrackCollection );

   /// Get the primary trajectories (produced by the Refitter)
   edm::Handle< TrajectoryCollection > primaryTrajectoryCollection;
   iEvent.getByLabel( "TrackRefitter", primaryTrajectoryCollection );

   /// Get the AssociationMap between primary tracks and trajectories
   edm::Handle< TrajTrackAssociationCollection > refMapH;
   iEvent.getByLabel( "TrackRefitter", refMapH );
   const TrajTrackAssociationCollection& refMap = *(refMapH.product());

   /// Get the AssociationMap between seeds and primary trajectories
   edm::Handle<TrajectoryToSeedsMap>  nuclMapH;
   iEvent.getByLabel("nuclearSeed", nuclMapH);
   const TrajectoryToSeedsMap& nuclMap = *(nuclMapH.product());

   /// Definition of the output AssociationMap
   std::auto_ptr<TrackToSeedsMap> outAssoc1(new TrackToSeedsMap);
   //std::auto_ptr<TrackToTracksMap> outAssoc2(new TrackToTracksMap);

   typedef edm::RefVector<TrajectorySeedCollection> TrajectorySeedRefVector;

   typedef edm::Ref<TrajectoryCollection> TrajectoryRef;

   /// Loop on all primary trajectories
   for(unsigned int i = 0; i < primaryTrajectoryCollection->size() ; i++) {

         TrajectoryRef  trajRef( primaryTrajectoryCollection, i );

         /// Get the primary track from the trajectory
         reco::TrackRef track = refMap[ trajRef ];
         if( track.isNull() ) {
                LogDebug("NuclearSeedGenerator") << "No tracks associated to the current trajectory by the TrackRefitter \n";
                continue;
        }

         /// Get the seeds from the trajectory
         try {
            TrajectorySeedRefVector seeds = nuclMap[ trajRef ];

            /// Insert the association between primary track and nuclear seeds
            for(TrajectorySeedRefVector::const_iterator it=seeds.begin(); it!=seeds.end(); it++) {
                   outAssoc1->insert(track, *it);
            }
         }
         catch ( edm::Exception event ) {
               LogDebug("NuclearSeedGenerator") << "Nuclear interaction identifed but no seeds found\n";
          }
   }

   iEvent.put(outAssoc1);
//   iEvent.put(outAssoc2);
}

// ------------ method called once each job just before starting event loop  ------------
void
NuclearSeedsToTrackAssociationEDProducer::beginJob(const edm::EventSetup& es)
{
}

void  NuclearSeedsToTrackAssociationEDProducer::endJob() {}
