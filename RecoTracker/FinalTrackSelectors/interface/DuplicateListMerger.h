#ifndef RecoTracker_DuplicateListMerger_h
#define RecoTracker_DuplicateListMerger_h
/** \class DuplicateListMerger
 * 
 * merges list of merge duplicate tracks with its parent list
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/EDProducer.h"
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
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>

#include "TMVA/Reader.h"

namespace reco { namespace modules {
    class DuplicateListMerger : public edm::EDProducer {
       public:
         /// constructor
         explicit DuplicateListMerger(const edm::ParameterSet& iPara);
	 /// destructor
	 virtual ~DuplicateListMerger();

	 /// typedef container of candidate and input tracks
	 typedef std::pair<TrackCandidate,std::pair<reco::Track,reco::Track> > DuplicateRecord;
	 typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
       protected:
	 /// produce one event
	 void produce( edm::Event &, const edm::EventSetup &);

       private:
	 int matchCandidateToTrack(TrackCandidate,edm::Handle<edm::View<reco::Track> >);

	 edm::ProductID clusterProductB( const TrackingRecHit *hit){
	   return reinterpret_cast<const BaseTrackerRecHit *>(hit)->firstClusterRef().id();
	 }

	 /// track input collection
	 edm::InputTag mergedTrackSource_;
	 edm::InputTag originalTrackSource_;
	 edm::InputTag candidateSource_;
	 reco::TrackBase::TrackQuality qualityToSet_;
	 unsigned int diffHitsCut_;
	 float minTrkProbCut_;
	 bool copyExtras_;
	 bool makeReKeyedSeeds_;
     };
  }
}
#endif
