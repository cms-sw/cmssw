#ifndef RecoTauTag_HLTProducers_PFJetsMaxInvMassModule_H
#define RecoTauTag_HLTProducers_PFJetsMaxInvMassModule_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

class PFJetsMaxInvMassModule : public edm::global::EDProducer<> {
private:
  const edm::EDGetTokenT<reco::PFJetCollection> pfJetSrc_;
  bool maxInvMassPairOnly_;
  bool removeMaxInvMassPair_;

public:
  explicit PFJetsMaxInvMassModule(const edm::ParameterSet&);
  ~PFJetsMaxInvMassModule() override = default;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};
#endif
