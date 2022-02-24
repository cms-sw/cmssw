#include "RecoTauTag/HLTProducers/interface/PFJetsTauCorrelationCondition.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
PFJetsTauCorrelationCondition::PFJetsTauCorrelationCondition(const edm::ParameterSet& iConfig)
    : tauSrc_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("TauSrc"))),
      pfJetSrc_(consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("PFJetSrc"))),
      extraTauPtCut_(iConfig.getParameter<double>("extraTauPtCut")),
      mjjMin_(iConfig.getParameter<double>("mjjMin")),
      matchingR2_(iConfig.getParameter<double>("Min_dR") * iConfig.getParameter<double>("Min_dR")) {
    produces<reco::PFJetCollection>();
}
PFJetsTauCorrelationCondition::~PFJetsTauCorrelationCondition() {}

void PFJetsTauCorrelationCondition::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
    std::unique_ptr<reco::PFJetCollection> cleanedPFJets(new reco::PFJetCollection);

    edm::Handle<trigger::TriggerFilterObjectWithRefs> tauJets;
    iEvent.getByToken(tauSrc_, tauJets);

    edm::Handle<reco::PFJetCollection> PFJets;
    iEvent.getByToken(pfJetSrc_, PFJets);

    trigger::VRpftau taus;
    tauJets->getObjects(trigger::TriggerTau, taus);

    std::set<unsigned int> indices; 

    if (PFJets->size() > 1 && taus.size() > 1) {
        for (unsigned int iJet1 = 0; iJet1 < PFJets->size(); iJet1++) {
            for (unsigned int iJet2 = iJet1+1; iJet2 < PFJets->size(); iJet2++) {
                bool correctComb = false;
                const reco::PFJet& myPFJet1 = (*PFJets)[iJet1];
                const reco::PFJet& myPFJet2 = (*PFJets)[iJet2];

                double mjj = (myPFJet1.p4() + myPFJet2.p4()).M();
                if (mjj < mjjMin_) continue;

                for (unsigned int iTau1 = 0; iTau1 < taus.size(); iTau1++) {
                    bool isMatched11 = (reco::deltaR2(taus[iTau1]->p4(), myPFJet1.p4()) < matchingR2_);
                    bool isMatched12 = (reco::deltaR2(taus[iTau1]->p4(), myPFJet2.p4()) < matchingR2_);
                    if (isMatched11 || isMatched12) continue;

                    for (unsigned int iTau2 = iTau1+1; iTau2 < taus.size(); iTau2++) {
                        bool isMatched21 = (reco::deltaR2(taus[iTau2]->p4(), myPFJet1.p4()) < matchingR2_);
                        bool isMatched22 = (reco::deltaR2(taus[iTau2]->p4(), myPFJet2.p4()) < matchingR2_);
                        if (isMatched21 || isMatched22) continue;

                        if (taus[iTau1]->pt() < extraTauPtCut_ && taus[iTau2]->pt() < extraTauPtCut_) continue;

                        correctComb = true;
                    }
                }

                if (correctComb == true) {
                    indices.insert(iJet1);
                    indices.insert(iJet2);
                }
            }
        }

        for (const auto& i : indices)
            cleanedPFJets->push_back((*PFJets)[i]);
    }
    iEvent.put(std::move(cleanedPFJets));
}

void PFJetsTauCorrelationCondition::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("PFJetSrc", edm::InputTag("hltAK4PFJetsCorrected"))->setComment("Input collection of PFJets");
    desc.add<edm::InputTag>("TauSrc", edm::InputTag("hltSinglePFTau20TrackPt1LooseChargedIsolationReg"))
      ->setComment("Input collection of PFTaus that have passed ID and isolation requirements");
    desc.add<double>("extraTauPtCut", 45)->setComment("In case of asymmetric tau pt cuts");
    desc.add<double>("mjjMin", 500)->setComment("VBF dijet mass condition");
    desc.add<double>("Min_dR", 0.5)->setComment("Minimum dR outside of which PFJets will be saved");
    descriptions.setComment(
      "This module produces a collection of PFJets that are cross-cleaned with respect to PFTaus passing a HLT "
      "filter.");
    descriptions.add("PFJetsTauCorrelationCondition", desc);
}
