#ifndef RecoJets_JetProducers_plugins_FixedGridRhoProducerFastjet_h
#define RecoJets_JetProducers_plugins_FixedGridRhoProducerFastjet_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"


class FixedGridRhoProducerFastjet : public edm::stream::EDProducer<> {

 public:
  explicit FixedGridRhoProducerFastjet(const edm::ParameterSet& iConfig);
  virtual ~FixedGridRhoProducerFastjet();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag pfCandidatesTag_;
  fastjet::GridMedianBackgroundEstimator bge_;

  edm::EDGetTokenT<edm::View<reco::Candidate> > input_pfcoll_token_;

};


#endif
