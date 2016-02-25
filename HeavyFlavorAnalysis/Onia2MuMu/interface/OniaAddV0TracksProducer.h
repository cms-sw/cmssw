#ifndef __OniaAddV0TracksProducer_h_
#define __OniaAddV0TracksProducer_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

#include <vector>

/**
   Add tracks from V0

 */

class OniaAddV0TracksProducer : public edm::EDProducer {

 public:
  explicit OniaAddV0TracksProducer(const edm::ParameterSet& ps);
 
 private:

  virtual void produce(edm::Event& event, const edm::EventSetup& esetup);
  virtual void endJob() ;

  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> LambdaCollectionToken_;
  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> KShortCollectionToken_;

  int events_v0;
  int total_v0;
  int total_lambda;
  int total_kshort;

};

#endif
