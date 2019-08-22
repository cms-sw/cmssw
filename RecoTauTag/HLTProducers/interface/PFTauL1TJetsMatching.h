#ifndef RecoTauTag_HLTProducers_PFTauL1TJetsMatching_H
#define RecoTauTag_HLTProducers_PFTauL1TJetsMatching_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

class PFTauL1TJetsMatching : public edm::global::EDProducer<> {
private:
  const edm::EDGetTokenT<reco::PFTauCollection> tauSrc_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> L1JetSrc_;
  const double matchingR2_;
  const double minTauPt_;
  const double minL1TPt_;

public:
  explicit PFTauL1TJetsMatching(const edm::ParameterSet&);
  ~PFTauL1TJetsMatching() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};
#endif
