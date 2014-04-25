#ifndef RecoTracker_DuplicateListMerger_h
#define RecoTracker_DuplicateListMerger_h
/** \class DuplicateListMerger
 * 
 * merges list of merge duplicate tracks with its parent list
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>

#include "TMVA/Reader.h"

namespace reco { namespace modules {
    class DuplicateListMerger : public edm::stream::EDProducer<> {
       public:
         /// constructor
         explicit DuplicateListMerger(const edm::ParameterSet& iPara);
	 /// destructor
	 virtual ~DuplicateListMerger();

	 /// typedef container of candidate and input tracks
	 typedef std::pair<TrackCandidate,std::pair<reco::TrackRef,reco::TrackRef> > DuplicateRecord;
	 typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
       protected:
	 /// produce one event
	 void produce( edm::Event &, const edm::EventSetup &) override;

       private:
	 int matchCandidateToTrack(TrackCandidate,edm::Handle<reco::TrackCollection>);

	 edm::ProductID clusterProductB( const TrackingRecHit *hit){
	   return reinterpret_cast<const BaseTrackerRecHit *>(hit)->firstClusterRef().id();
	 }

	 /// track input collection
         struct ThreeTokens {
            edm::InputTag tag;
            edm::EDGetTokenT<reco::TrackCollection> tk;
            edm::EDGetTokenT<std::vector<Trajectory> >        traj;
            edm::EDGetTokenT<TrajTrackAssociationCollection > tass;
            ThreeTokens() {}
            ThreeTokens(const edm::InputTag &tag_, edm::EDGetTokenT<reco::TrackCollection> && tk_, edm::EDGetTokenT<std::vector<Trajectory> > && traj_, edm::EDGetTokenT<TrajTrackAssociationCollection > && tass_) :
                tag(tag_), tk(tk_), traj(traj_), tass(tass_) {}
         };
         ThreeTokens threeTokens(const edm::InputTag &tag) {
            return ThreeTokens(tag, consumes<reco::TrackCollection>(tag), consumes<std::vector<Trajectory> >(tag), consumes<TrajTrackAssociationCollection >(tag));
         }
         ThreeTokens mergedTrackSource_, originalTrackSource_;
         edm::EDGetTokenT<edm::View<DuplicateRecord> > candidateSource_;

         edm::InputTag originalMVAVals_;
         edm::InputTag mergedMVAVals_;
         edm::EDGetTokenT<edm::ValueMap<float> > originalMVAValsToken_;
         edm::EDGetTokenT<edm::ValueMap<float> > mergedMVAValsToken_;

	 reco::TrackBase::TrackQuality qualityToSet_;
	 unsigned int diffHitsCut_;
	 float minTrkProbCut_;
	 bool copyExtras_;
	 bool makeReKeyedSeeds_;
     };
  }
}
#endif
