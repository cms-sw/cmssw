#include "HLTrigger/JetMET/interface/HLTJetL1MatchProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HLTJetL1MatchProducer::HLTJetL1MatchProducer(const edm::ParameterSet& iConfig)
{
  jetsInput_ = iConfig.getParameter<edm::InputTag>("jetsInput");
  L1TauJets_ = iConfig.getParameter<edm::InputTag>("L1TauJets");
  L1CenJets_ = iConfig.getParameter<edm::InputTag>("L1CenJets");
  L1ForJets_ = iConfig.getParameter<edm::InputTag>("L1ForJets");
  DeltaR_ = iConfig.getParameter<double>("DeltaR");

  produces< reco::CaloJetCollection > ();

}

void HLTJetL1MatchProducer::beginJob()
{

}

HLTJetL1MatchProducer::~HLTJetL1MatchProducer()
{

}

void HLTJetL1MatchProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::CaloJetCollection> calojets;
  iEvent.getByLabel(jetsInput_, calojets);

  std::auto_ptr<reco::CaloJetCollection> result (new reco::CaloJetCollection);


  edm::Handle<l1extra::L1JetParticleCollection> l1TauJets;
  iEvent.getByLabel(L1TauJets_,l1TauJets);

  edm::Handle<l1extra::L1JetParticleCollection> l1CenJets;
  iEvent.getByLabel(L1CenJets_,l1CenJets);

  edm::Handle<l1extra::L1JetParticleCollection> l1ForJets;
  iEvent.getByLabel(L1ForJets_,l1ForJets);


  for (reco::CaloJetCollection::const_iterator calojetc = calojets->begin(); 
       calojetc != calojets->end(); ++calojetc) {

    bool isMatched=false;

    //std::cout << "FL: l1TauJets.size  = " << l1TauJets->size() << std::endl;
    for (unsigned int jetc=0;jetc<l1TauJets->size();++jetc)
    {
      const double deltaeta=calojetc->eta()-(*l1TauJets)[jetc].eta();
      const double deltaphi=deltaPhi(calojetc->phi(),(*l1TauJets)[jetc].phi());
      //std::cout << "FL: sqrt(2) = " << sqrt(2) << std::endl;
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }

    for (unsigned int jetc=0;jetc<l1CenJets->size();++jetc)
    {
      const double deltaeta=calojetc->eta()-(*l1CenJets)[jetc].eta();
      const double deltaphi=deltaPhi(calojetc->phi(),(*l1CenJets)[jetc].phi());
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }

    for (unsigned int jetc=0;jetc<l1ForJets->size();++jetc)
    {
      const double deltaeta=calojetc->eta()-(*l1ForJets)[jetc].eta();
      const double deltaphi=deltaPhi(calojetc->phi(),(*l1ForJets)[jetc].phi());
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }


    if (isMatched==true) result->push_back(*calojetc);

  } // calojetc

  iEvent.put( result);

}


