#ifndef __OniaAddV0TracksProducer_h_
#define __OniaAddV0TracksProducer_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

#include <vector>
#include <atomic>

/**
   Add tracks from V0

 */

class OniaAddV0TracksProducer : public edm::global::EDProducer<> {
public:
  explicit OniaAddV0TracksProducer(const edm::ParameterSet& ps);

private:
  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup& esetup) const override;
  void endJob() override;

  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> LambdaCollectionToken_;
  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> KShortCollectionToken_;

  mutable std::atomic<int> events_v0;
  mutable std::atomic<int> total_v0;
  mutable std::atomic<int> total_lambda;
  mutable std::atomic<int> total_kshort;
};

#endif
