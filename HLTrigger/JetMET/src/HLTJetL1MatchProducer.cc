#include <string>

#include "HLTrigger/JetMET/interface/HLTJetL1MatchProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template<typename T>
HLTJetL1MatchProducer<T>::HLTJetL1MatchProducer(const edm::ParameterSet& iConfig)
{
  jetsInput_ = iConfig.template getParameter<edm::InputTag>("jetsInput");
  L1TauJets_ = iConfig.template getParameter<edm::InputTag>("L1TauJets");
  L1CenJets_ = iConfig.template getParameter<edm::InputTag>("L1CenJets");
  L1ForJets_ = iConfig.template getParameter<edm::InputTag>("L1ForJets");
  DeltaR_ = iConfig.template getParameter<double>("DeltaR");

  typedef std::vector<T> TCollection;
  m_theJetToken = consumes<TCollection>(jetsInput_);
  m_theL1TauJetToken = consumes<l1extra::L1JetParticleCollection>(L1TauJets_);
  m_theL1CenJetToken = consumes<l1extra::L1JetParticleCollection>(L1CenJets_);
  m_theL1ForJetToken = consumes<l1extra::L1JetParticleCollection>(L1ForJets_);
  produces<TCollection> ();

}

template<typename T>
void HLTJetL1MatchProducer<T>::beginJob()
{

}

template<typename T>
HLTJetL1MatchProducer<T>::~HLTJetL1MatchProducer()
{

}

template<typename T>
void HLTJetL1MatchProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput",edm::InputTag("hltAntiKT5PFJets"));
  desc.add<edm::InputTag>("L1TauJets",edm::InputTag("hltL1extraParticles","Tau"));
  desc.add<edm::InputTag>("L1CenJets",edm::InputTag("hltL1extraParticles","Central"));
  desc.add<edm::InputTag>("L1ForJets",edm::InputTag("hltL1extraParticles","Forward"));
  desc.add<double>("DeltaR",0.5);
  descriptions.add(defaultModuleLabel<HLTJetL1MatchProducer<T>>(), desc);
}

template<typename T>
void HLTJetL1MatchProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::vector<T> TCollection;

  edm::Handle<TCollection> jets;
  iEvent.getByToken(m_theJetToken, jets);

  std::auto_ptr<TCollection> result (new TCollection);


  edm::Handle<l1extra::L1JetParticleCollection> l1TauJets;
  iEvent.getByToken(m_theL1TauJetToken,l1TauJets);

  edm::Handle<l1extra::L1JetParticleCollection> l1CenJets;
  iEvent.getByToken(m_theL1CenJetToken,l1CenJets);

  edm::Handle<l1extra::L1JetParticleCollection> l1ForJets;
  iEvent.getByToken(m_theL1ForJetToken,l1ForJets);

  typename TCollection::const_iterator jet_iter;
  for (jet_iter = jets->begin(); jet_iter != jets->end(); ++jet_iter) {

    bool isMatched=false;

    //std::cout << "FL: l1TauJets.size  = " << l1TauJets->size() << std::endl;
    for (unsigned int jetc=0;jetc<l1TauJets->size();++jetc)
    {
      const double deltaeta=jet_iter->eta()-(*l1TauJets)[jetc].eta();
      const double deltaphi=deltaPhi(jet_iter->phi(),(*l1TauJets)[jetc].phi());
      //std::cout << "FL: sqrt(2) = " << sqrt(2) << std::endl;
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }

    for (unsigned int jetc=0;jetc<l1CenJets->size();++jetc)
    {
      const double deltaeta=jet_iter->eta()-(*l1CenJets)[jetc].eta();
      const double deltaphi=deltaPhi(jet_iter->phi(),(*l1CenJets)[jetc].phi());
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }

    for (unsigned int jetc=0;jetc<l1ForJets->size();++jetc)
    {
      const double deltaeta=jet_iter->eta()-(*l1ForJets)[jetc].eta();
      const double deltaphi=deltaPhi(jet_iter->phi(),(*l1ForJets)[jetc].phi());
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }


    if (isMatched==true) result->push_back(*jet_iter);

  } // jet_iter

  iEvent.put( result);

}


