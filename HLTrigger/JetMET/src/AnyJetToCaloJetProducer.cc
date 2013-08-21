#include "HLTrigger/JetMET/interface/AnyJetToCaloJetProducer.h"
#include "DataFormats/Common/interface/Handle.h"
 
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h" 

AnyJetToCaloJetProducer::AnyJetToCaloJetProducer(const edm::ParameterSet& iConfig)
{
  jetSrc_ = iConfig.getParameter<edm::InputTag>("Source");
  m_theGenericJetToken = consumes<edm::View<reco::Jet>>(jetSrc_);
  produces<reco::CaloJetCollection>();
}

AnyJetToCaloJetProducer::~AnyJetToCaloJetProducer(){ }

void
AnyJetToCaloJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Source",edm::InputTag(""));
  descriptions.add("AnyJetToCaloJetProducer",desc);
}

void AnyJetToCaloJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  std::auto_ptr<reco::CaloJetCollection> newjets(new reco::CaloJetCollection());
  
  edm::Handle<edm::View<reco::Jet> > jets;
  if(iEvent.getByToken(m_theGenericJetToken,jets)) {
    for(edm::View<reco::Jet>::const_iterator i = jets->begin(); i != jets->end(); i++ ) {
      reco::CaloJet jet(i->p4(), i->vertex(), reco::CaloJet::Specific());
      newjets->push_back(jet);
    }
  }
  
  iEvent.put(newjets);
}


