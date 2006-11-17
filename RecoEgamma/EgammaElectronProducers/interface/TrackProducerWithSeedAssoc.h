#ifndef TrackProducerWithSeedAssoc_h
#define TrackProducerWithSeedAssoc_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducerWithSeedAssoc
// 
//
// Description: Produce Tracks from TrackCandidates 
// write Associationmap tracks-seeds at the same time
//
//
// Original Author:  Ursula Berthon, Claude Charlotk
// adaptation from TrackProducer from Giuseppe Cerati, just adding associationmap
//         Created:  Thu Nov  9 17:29:31 CET 2006
//

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class TrackProducerWithSeedAssoc : public TrackProducerBase, public edm::EDProducer {
public:

  explicit TrackProducerWithSeedAssoc(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  TrackProducerAlgorithm theAlgo;
  std::string assocModule_;
  edm::OrphanHandle<reco::TrackCollection> rTracks_;
  bool myTrajectoryInEvent_;

  //we had to copy this from TrackProducerBase to get the OrphanHandle
  //ugly temporary solution!!
  void putInEvt(edm::Event& evt,
				 std::auto_ptr<TrackingRecHitCollection>& selHits,
				 std::auto_ptr<reco::TrackCollection>& selTracks,
				 std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
				 std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
					      AlgoProductCollection& algoResults);

};

#endif
