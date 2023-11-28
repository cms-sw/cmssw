#include "RecoTauTag/HLTProducers/interface/L1THLTTauMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/TauReco/interface/PFTau.h"

//
// class declaration
//
using namespace reco;
using namespace std;
using namespace edm;
using namespace trigger;

L1THLTTauMatching::L1THLTTauMatching(const edm::ParameterSet& iConfig)
    : jetSrc(consumes<PFTauCollection>(iConfig.getParameter<InputTag>("JetSrc"))),
      tauTrigger(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1TauTrigger"))),
      mEt_Min(iConfig.getParameter<double>("EtMin")),
      reduceTauContent(iConfig.getParameter<bool>("ReduceTauContent")),
      keepOriginalVertex(iConfig.getParameter<bool>("KeepOriginalVertex")) {
  produces<PFTauCollection>();
}

void L1THLTTauMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  unique_ptr<PFTauCollection> tauL2jets(new PFTauCollection);

  constexpr double matchingR2 = 0.5 * 0.5;

  // Getting HLT jets to be matched
  edm::Handle<PFTauCollection> tauJets;
  iEvent.getByToken(jetSrc, tauJets);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
  iEvent.getByToken(tauTrigger, l1TriggeredTaus);

  l1t::TauVectorRef tauCandRefVec;
  l1TriggeredTaus->getObjects(trigger::TriggerL1Tau, tauCandRefVec);

  math::XYZPoint a(0., 0., 0.);

  for (unsigned int iL1Tau = 0; iL1Tau < tauCandRefVec.size(); iL1Tau++) {
    for (unsigned int iJet = 0; iJet < tauJets->size(); iJet++) {
      // Find the relative L2TauJets, to see if it has been reconstructed
      const PFTau& myJet = (*tauJets)[iJet];
      double deltaR2 = ROOT::Math::VectorUtil::DeltaR2(myJet.p4().Vect(), (tauCandRefVec[iL1Tau]->p4()).Vect());
      if (deltaR2 < matchingR2) {
        if (myJet.leadChargedHadrCand().isNonnull()) {
          a = myJet.leadChargedHadrCand()->vertex();
        }

        auto myPFTau =
            reduceTauContent ? PFTau(std::numeric_limits<int>::quiet_NaN(), myJet.p4(), myJet.vertex()) : PFTau(myJet);

        if (!keepOriginalVertex) {
          myPFTau.setVertex(a);
        }

        if (myPFTau.pt() > mEt_Min) {
          tauL2jets->push_back(myJet);
        }
        break;
      }
    }
  }

  iEvent.put(std::move(tauL2jets));
}

void L1THLTTauMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1TauTrigger", edm::InputTag("hltL1sDoubleIsoTau40er"))
      ->setComment("Name of trigger filter");
  desc.add<edm::InputTag>("JetSrc", edm::InputTag("hltSelectedPFTausTrackPt1MediumIsolationReg"))
      ->setComment("Input collection of PFTaus");
  desc.add<double>("EtMin", 0.0)->setComment("Minimal pT of PFTau to match");
  desc.add<bool>("ReduceTauContent", true)->setComment("Should produce taus with reduced content (Only p4 and vertex)");
  desc.add<bool>("KeepOriginalVertex", false)
      ->setComment("Should use original vertex instead of setting the vertex to that of the leading charged hadron");
  descriptions.setComment("This module produces collection of PFTaus matched to L1 Taus / Jets passing a HLT filter.");
  descriptions.add("L1THLTTauMatching", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1THLTTauMatching);
