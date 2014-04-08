#ifndef RecoJets_JetProducers_BasicToPFJet_h
#define RecoJets_JetProducers_BasicToPFJet_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/JetReco/interface/BasicJet.h"


class BasicToPFJet : public edm::EDProducer {

 public:

  explicit BasicToPFJet(const edm::ParameterSet& PSet);
  virtual ~BasicToPFJet();
  virtual void produce(edm::Event & event, const edm::EventSetup & EventSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  edm::InputTag src_;
  const edm::EDGetTokenT<reco::BasicJetCollection> inputToken_;
};


#endif
