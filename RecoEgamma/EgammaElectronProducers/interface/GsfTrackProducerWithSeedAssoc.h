#ifndef GsfTrackProducerWithSeedAssoc_h
#define GsfTrackProducerWithSeedAssoc_h

//
// Package:    RecoEgamma/EgammaElectronProducers
// Class:      GsfTrackProducerWithSeedAssoc
// 
//
// Description: Produce GsfTracks from TrackCandidates 
// write Associationmap tracks-seeds at the same time
//
//
// Original Author:  Ursula Berthon, Claude Charlotk
// close addaptation from GsfTrackProducer from Giuseppe Cerati, just adding associationmap
//         Created:  Thu Nov  9 17:29:31 CET 2006
//

#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"

//#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/GsfTrackFwd.h"

class GsfTrackProducerWithSeedAssoc : public GsfTrackProducerBase, public edm::EDProducer {
public:

  explicit GsfTrackProducerWithSeedAssoc(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

  //  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  GsfTrackProducerAlgorithm theAlgo;
  std::string assocModule_;
  edm::OrphanHandle<reco::GsfTrackCollection> rTracks_;
  bool myTrajectoryInEvent_;
  //  std::string assocProduct_;
  //we had to copy this from TrackProducerBase to get the OrphanHandle
  //ugly temporary solution!!
  void putInEvt(edm::Event& evt,
				 std::auto_ptr<TrackingRecHitCollection>& selHits,
				 std::auto_ptr<reco::GsfTrackCollection>& selTracks,
				 std::auto_ptr<reco::GsfTrackExtraCollection>& selTrackExtras,
				 std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
					      AlgoProductCollection& algoResults);

};

#endif
