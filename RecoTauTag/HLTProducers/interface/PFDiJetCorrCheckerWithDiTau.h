#ifndef RecoTauTag_HLTProducers_PFDiJetCorrCheckerWithDiTau_H
#define RecoTauTag_HLTProducers_PFDiJetCorrCheckerWithDiTau_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

class PFDiJetCorrCheckerWithDiTau : public edm::stream::EDProducer<> {
private:
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauSrc_;
  const edm::EDGetTokenT<reco::PFJetCollection> pfJetSrc_;
  const double extraTauPtCut_;
  const double mjjMin_;
  const double matchingR2_;

public:
  explicit PFDiJetCorrCheckerWithDiTau(const edm::ParameterSet&);
  ~PFDiJetCorrCheckerWithDiTau() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};
#endif
