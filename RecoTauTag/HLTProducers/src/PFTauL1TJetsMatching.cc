#include "RecoTauTag/HLTProducers/interface/PFTauL1TJetsMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
PFTauL1TJetsMatching::PFTauL1TJetsMatching(const edm::ParameterSet& iConfig)
    : tauSrc_(consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>("TauSrc"))),
      L1JetSrc_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L1JetSrc"))),
      matchingR2_(iConfig.getParameter<double>("MatchingdR") * iConfig.getParameter<double>("MatchingdR")),
      minTauPt_(iConfig.getParameter<double>("MinTauPt")),
      minL1TPt_(iConfig.getParameter<double>("MinL1TPt")) {
  produces<reco::PFTauCollection>();
}
PFTauL1TJetsMatching::~PFTauL1TJetsMatching() {}

void PFTauL1TJetsMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  std::unique_ptr<reco::PFTauCollection> L1TmatchedPFTau(new reco::PFTauCollection);

  edm::Handle<reco::PFTauCollection> taus;
  iEvent.getByToken(tauSrc_, taus);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1Jets;
  iEvent.getByToken(L1JetSrc_, L1Jets);

  l1t::JetVectorRef jetCandRefVec;
  L1Jets->getObjects(trigger::TriggerL1Jet, jetCandRefVec);

  /* Loop over taus that must pass a certain minTauPt_ cut */
  /* then loop over L1T jets that must pass minL1TPt_ */
  /* and check whether they match, if yes -> include the taus in */
  /* the new L1T matched PFTau collection */

  for (unsigned int iTau = 0; iTau < taus->size(); iTau++) {
    bool isMatched = false;
    if ((*taus)[iTau].pt() > minTauPt_) {
      for (unsigned int iJet = 0; iJet < jetCandRefVec.size(); iJet++) {
        if (jetCandRefVec[iJet]->pt() > minL1TPt_) {
          if (reco::deltaR2((*taus)[iTau].p4(), jetCandRefVec[iJet]->p4()) < matchingR2_) {
            isMatched = true;
            break;
          }
        }
      }
    }
    if (isMatched)
      L1TmatchedPFTau->push_back((*taus)[iTau]);
  }
  iEvent.put(std::move(L1TmatchedPFTau));
}

void PFTauL1TJetsMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1JetSrc", edm::InputTag("hltL1VBFDiJetOR"))
      ->setComment("Input filter objects passing L1 seed");
  desc.add<edm::InputTag>("TauSrc", edm::InputTag("hltSelectedPFTausTrackFindingLooseChargedIsolationAgainstMuon"))
      ->setComment("Input collection of PFTaus");
  desc.add<double>("MatchingdR", 0.5)->setComment("Maximum dR for matching between PFTaus and L1 filter jets");
  desc.add<double>("MinTauPt", 20.0)->setComment("PFTaus above this pt will be considered");
  desc.add<double>("MinL1TPt", 115.0)->setComment("L1T Objects above this pt will be considered");
  descriptions.setComment(
      "This module produces a collection of PFTaus matched to the leading jet passing the L1 seed filter.");
  descriptions.add("PFTauL1TJetsMatching", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTauL1TJetsMatching);
