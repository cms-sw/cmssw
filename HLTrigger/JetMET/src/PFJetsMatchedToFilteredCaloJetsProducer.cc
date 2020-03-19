#include "HLTrigger/JetMET/interface/PFJetsMatchedToFilteredCaloJetsProducer.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;

PFJetsMatchedToFilteredCaloJetsProducer::PFJetsMatchedToFilteredCaloJetsProducer(const edm::ParameterSet& iConfig) {
  PFJetSrc = iConfig.getParameter<InputTag>("PFJetSrc");
  CaloJetFilter = iConfig.getParameter<InputTag>("CaloJetFilter");
  DeltaR_ = iConfig.getParameter<double>("DeltaR");
  TriggerType_ = iConfig.getParameter<int>("TriggerType");

  m_thePFJetToken = consumes<edm::View<reco::Candidate> >(PFJetSrc);
  m_theTriggerJetToken = consumes<trigger::TriggerFilterObjectWithRefs>(CaloJetFilter);

  produces<PFJetCollection>();
}

PFJetsMatchedToFilteredCaloJetsProducer::~PFJetsMatchedToFilteredCaloJetsProducer() = default;

void PFJetsMatchedToFilteredCaloJetsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFJetSrc", edm::InputTag("hltPFJets"));
  desc.add<edm::InputTag>("CaloJetFilter", edm::InputTag("hltSingleJet240Regional"));
  desc.add<double>("DeltaR", 0.5);
  desc.add<int>("TriggerType", trigger::TriggerJet);
  descriptions.add("hltPFJetsMatchedToFilteredCaloJetsProducer", desc);
}

void PFJetsMatchedToFilteredCaloJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES) {
  using namespace edm;
  using namespace std;
  using namespace reco;
  using namespace trigger;

  unique_ptr<PFJetCollection> pfjets(new PFJetCollection);

  //Getting HLT jets to be matched
  edm::Handle<edm::View<reco::Candidate> > PFJets;
  iEvent.getByToken(m_thePFJetToken, PFJets);

  //std::cout <<"Size of input PF jet collection "<<PFJets->size()<<std::endl;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> TriggeredCaloJets;
  iEvent.getByToken(m_theTriggerJetToken, TriggeredCaloJets);

  jetRefVec.clear();

  TriggeredCaloJets->getObjects(TriggerType_, jetRefVec);
  // std::cout <<"Size of input triggered jet collection "<<jetRefVec.size()<<std::endl;
  math::XYZPoint a(0., 0., 0.);
  PFJet::Specific f;

  for (auto& iCalo : jetRefVec) {
    // std::cout << "\tiTriggerJet: " << iCalo << " pT= " << jetRefVec[iCalo]->pt() << std::endl;
    for (unsigned int iPF = 0; iPF < PFJets->size(); iPF++) {
      const Candidate& myJet = (*PFJets)[iPF];
      double deltaR = ROOT::Math::VectorUtil::DeltaR(myJet.p4().Vect(), (iCalo->p4()).Vect());
      if (deltaR < DeltaR_) {
        PFJet myPFJet(myJet.p4(), a, f);
        pfjets->push_back(myPFJet);
        break;
      }
    }
  }

  // std::cout <<"Size of PF matched jets "<<pfjets->size()<<std::endl;
  iEvent.put(std::move(pfjets));
}
