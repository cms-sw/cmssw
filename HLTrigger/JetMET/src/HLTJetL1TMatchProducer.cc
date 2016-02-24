#include <string>

#include "HLTrigger/JetMET/interface/HLTJetL1TMatchProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template<typename T>
HLTJetL1TMatchProducer<T>::HLTJetL1TMatchProducer(const edm::ParameterSet& iConfig)
{
  jetsInput_ = iConfig.template getParameter<edm::InputTag>("jetsInput");
  L1Jets_ = iConfig.template getParameter<edm::InputTag>("L1Jets");
  DeltaR_ = iConfig.template getParameter<double>("DeltaR");

  typedef std::vector<T> TCollection;
  m_theJetToken = consumes<TCollection>(jetsInput_);
  m_theL1JetToken = consumes<l1t::JetBxCollection>(L1Jets_);
  produces<TCollection> ();

}

template<typename T>
void HLTJetL1TMatchProducer<T>::beginJob()
{

}

template<typename T>
HLTJetL1TMatchProducer<T>::~HLTJetL1TMatchProducer()
{

}

template<typename T>
void HLTJetL1TMatchProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput",edm::InputTag("hltAntiKT5PFJets"));
  desc.add<edm::InputTag>("L1Jets",edm::InputTag("hltCaloStage2Digis"));
  desc.add<double>("DeltaR",0.5);
  descriptions.add(defaultModuleLabel<HLTJetL1TMatchProducer<T>>(), desc);
}

template<typename T>
void HLTJetL1TMatchProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::vector<T> TCollection;

  edm::Handle<TCollection> jets;
  iEvent.getByToken(m_theJetToken, jets);

  std::auto_ptr<TCollection> result (new TCollection);


  edm::Handle<l1t::JetBxCollection> l1Jets;
  iEvent.getByToken(m_theL1JetToken,l1Jets);

  typename TCollection::const_iterator jet_iter;
  for (jet_iter = jets->begin(); jet_iter != jets->end(); ++jet_iter) {

    bool isMatched=false;

    for (unsigned int jetc=0;jetc<l1Jets->size();++jetc)
    {
      const double deltaeta=jet_iter->eta()-(*l1Jets)[jetc].eta();
      const double deltaphi=deltaPhi(jet_iter->phi(),(*l1Jets)[jetc].phi());
      if (sqrt(deltaeta*deltaeta+deltaphi*deltaphi) < DeltaR_) isMatched=true;
    }


    if (isMatched==true) result->push_back(*jet_iter);

  } // jet_iter

  iEvent.put( result);

}


