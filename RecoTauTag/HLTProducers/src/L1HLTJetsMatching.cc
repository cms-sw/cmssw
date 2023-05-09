#include "RecoTauTag/HLTProducers/interface/L1HLTJetsMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
using namespace l1extra;

L1HLTJetsMatching::L1HLTJetsMatching(const edm::ParameterSet& iConfig) {
  jetSrc = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<InputTag>("JetSrc"));
  tauTrigger = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1TauTrigger"));
  mEt_Min = iConfig.getParameter<double>("EtMin");

  produces<CaloJetCollection>();
}

void L1HLTJetsMatching::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iES) const {
  using namespace edm;
  using namespace std;
  using namespace reco;
  using namespace trigger;
  using namespace l1extra;

  typedef std::vector<LeafCandidate> LeafCandidateCollection;

  unique_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection);

  constexpr double matchingR2 = 0.5 * 0.5;

  //Getting HLT jets to be matched
  edm::Handle<edm::View<Candidate> > tauJets;
  iEvent.getByToken(jetSrc, tauJets);

  //		std::cout <<"Size of input jet collection "<<tauJets->size()<<std::endl;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
  iEvent.getByToken(tauTrigger, l1TriggeredTaus);

  std::vector<l1extra::L1JetParticleRef> tauCandRefVec;
  std::vector<l1extra::L1JetParticleRef> jetCandRefVec;

  l1TriggeredTaus->getObjects(trigger::TriggerL1TauJet, tauCandRefVec);
  l1TriggeredTaus->getObjects(trigger::TriggerL1CenJet, jetCandRefVec);
  math::XYZPoint a(0., 0., 0.);
  CaloJet::Specific f;

  for (unsigned int iL1Tau = 0; iL1Tau < tauCandRefVec.size(); iL1Tau++) {
    for (unsigned int iJet = 0; iJet < tauJets->size(); iJet++) {
      //Find the relative L2TauJets, to see if it has been reconstructed
      const Candidate& myJet = (*tauJets)[iJet];
      double deltaR2 = ROOT::Math::VectorUtil::DeltaR2(myJet.p4().Vect(), (tauCandRefVec[iL1Tau]->p4()).Vect());
      if (deltaR2 < matchingR2) {
        //		 LeafCandidate myLC(myJet);
        CaloJet myCaloJet(myJet.p4(), a, f);
        if (myJet.pt() > mEt_Min) {
          //		  tauL2LC->push_back(myLC);
          tauL2jets->push_back(myCaloJet);
        }
        break;
      }
    }
  }

  for (unsigned int iL1Tau = 0; iL1Tau < jetCandRefVec.size(); iL1Tau++) {
    for (unsigned int iJet = 0; iJet < tauJets->size(); iJet++) {
      const Candidate& myJet = (*tauJets)[iJet];
      //Find the relative L2TauJets, to see if it has been reconstructed
      double deltaR2 = ROOT::Math::VectorUtil::DeltaR2(myJet.p4().Vect(), (jetCandRefVec[iL1Tau]->p4()).Vect());
      if (deltaR2 < matchingR2) {
        //		 LeafCandidate myLC(myJet);
        CaloJet myCaloJet(myJet.p4(), a, f);
        if (myJet.pt() > mEt_Min) {
          //tauL2LC->push_back(myLC);
          tauL2jets->push_back(myCaloJet);
        }
        break;
      }
    }
  }

  //std::cout <<"Size of L1HLT matched jets "<<tauL2jets->size()<<std::endl;

  iEvent.put(std::move(tauL2jets));
  // iEvent.put(std::move(tauL2LC));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1HLTJetsMatching);
