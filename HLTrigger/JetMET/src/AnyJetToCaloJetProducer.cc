
#include "HLTrigger/JetMET/interface/AnyJetToCaloJetProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

AnyJetToCaloJetProducer::AnyJetToCaloJetProducer(const edm::ParameterSet& iConfig)
{
  jetSrc_ = iConfig.getParameter<edm::InputTag>("Source");
  produces<reco::CaloJetCollection>();
}

AnyJetToCaloJetProducer::~AnyJetToCaloJetProducer(){ }

void AnyJetToCaloJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  std::auto_ptr<reco::CaloJetCollection> newjets(new reco::CaloJetCollection());

  edm::Handle<edm::View<reco::Jet> > jets;
  if (iEvent.getByLabel( jetSrc_, jets )) {
    for(edm::View<reco::Jet>::const_iterator i = jets->begin(); i != jets->end(); i++ ) {
      reco::CaloJet jet(i->p4(), i->vertex(), reco::CaloJet::Specific());
      newjets->push_back(jet);
    }
  }

  iEvent.put(newjets);

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AnyJetToCaloJetProducer);
