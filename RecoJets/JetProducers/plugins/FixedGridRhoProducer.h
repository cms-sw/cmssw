#ifndef RecoJets_JetProducers_plugins_FixedGridRhoProducer_h
#define RecoJets_JetProducers_plugins_FixedGridRhoProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoJets/JetAlgorithms/interface/FixedGridEnergyDensity.h"

class FixedGridRhoProducer : public edm::stream::EDProducer<> {

 public:
  explicit FixedGridRhoProducer(const edm::ParameterSet& iConfig);
  virtual ~FixedGridRhoProducer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag pfCandidatesTag_;
  FixedGridEnergyDensity::EtaRegion myEtaRegion;
  FixedGridEnergyDensity* algo; 

  edm::EDGetTokenT<reco::PFCandidateCollection> input_pfcoll_token_;

};


#endif
