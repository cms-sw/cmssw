#include "RecoTauTag/HLTProducers/interface/PFJetsTauOverlapRemoval.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
PFJetsTauOverlapRemoval::PFJetsTauOverlapRemoval(const edm::ParameterSet& iConfig)
    : tauSrc_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("TauSrc"))),
      pfJetSrc_(consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("PFJetSrc"))),
      matchingR2_(iConfig.getParameter<double>("Min_dR") * iConfig.getParameter<double>("Min_dR")) {
  produces<reco::PFJetCollection>();
}
PFJetsTauOverlapRemoval::~PFJetsTauOverlapRemoval() {}

void PFJetsTauOverlapRemoval::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  std::unique_ptr<reco::PFJetCollection> cleanedPFJets(new reco::PFJetCollection);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> tauJets;
  iEvent.getByToken(tauSrc_, tauJets);

  edm::Handle<reco::PFJetCollection> PFJets;
  iEvent.getByToken(pfJetSrc_, PFJets);

  trigger::VRpftau taus;
  tauJets->getObjects(trigger::TriggerTau, taus);

  if (PFJets->size() > 1) {
    for (unsigned int iJet = 0; iJet < PFJets->size(); iJet++) {
      bool isMatched = false;
      const reco::PFJet& myPFJet = (*PFJets)[iJet];
      for (unsigned int iTau = 0; iTau < taus.size(); iTau++) {
        if (reco::deltaR2(taus[iTau]->p4(), myPFJet.p4()) < matchingR2_) {
          isMatched = true;
          break;
        }
      }
      if (isMatched == false)
        cleanedPFJets->push_back(myPFJet);
    }
  }
  iEvent.put(std::move(cleanedPFJets));
}

void PFJetsTauOverlapRemoval::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFJetSrc", edm::InputTag("hltAK4PFJetsCorrected"))->setComment("Input collection of PFJets");
  desc.add<edm::InputTag>("TauSrc", edm::InputTag("hltSinglePFTau20TrackPt1LooseChargedIsolationReg"))
      ->setComment("Input collection of PFTaus that have passed ID and isolation requirements");
  desc.add<double>("Min_dR", 0.5)->setComment("Minimum dR outside of which PFJets will be saved");
  descriptions.setComment(
      "This module produces a collection of PFJets that are cross-cleaned with respect to PFTaus passing a HLT "
      "filter.");
  descriptions.add("PFJetsTauOverlapRemoval", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetsTauOverlapRemoval);
