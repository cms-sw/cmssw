#ifndef TrackInfoProducer_h
#define TrackInfoProducer_h

//
// Package:    RecoTracker/TrackInfoProducer
// Class:      TrackInfoProducer
//
//
// Description: Produce TrackInfo from Trajectory
//
//
// Original Author:  Chiara Genta
//         Created:
//

#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducerAlgorithm.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class TrackInfoProducer : public edm::EDProducer {
 public:

  //  TrackInfoProducer(){}
  explicit TrackInfoProducer(const edm::ParameterSet& iConfig);

  virtual ~TrackInfoProducer(){};

  virtual void produce(edm::Event&, const edm::EventSetup&);


 private:
  TrackInfoProducerAlgorithm theAlgo_;
  edm::EDGetTokenT<std::vector<Trajectory> > TrajectoryToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackCollectionToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> assoMapToken_;
  std::string forwardPredictedStateTag_, backwardPredictedStateTag_, updatedStateTag_, combinedStateTag_;
};
#endif
