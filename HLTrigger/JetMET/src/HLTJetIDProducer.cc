#include "HLTrigger/JetMET/interface/HLTJetIDProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

HLTJetIDProducer::HLTJetIDProducer(const edm::ParameterSet& iConfig)
{
  jetsInput_ = iConfig.getParameter<edm::InputTag>("jetsInput");
  min_EMF_ = iConfig.getParameter<double>("min_EMF");
  max_EMF_ = iConfig.getParameter<double>("max_EMF");
  min_N90_ = iConfig.getParameter<int>("min_N90");


  //  produces< reco::CaloJetCollection > ( "hltJetIDCollection" );
  produces< reco::CaloJetCollection > ();
}

void HLTJetIDProducer::beginJob()
{

}

HLTJetIDProducer::~HLTJetIDProducer()
{

}

void HLTJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::CaloJetCollection> calojets;
  iEvent.getByLabel(jetsInput_, calojets);

  std::auto_ptr<reco::CaloJetCollection> result (new reco::CaloJetCollection);

  for (reco::CaloJetCollection::const_iterator calojetc = calojets->begin(); 
       calojetc != calojets->end(); ++calojetc) {
      
    if (fabs(calojetc->eta()) >= 2.6) {
      result->push_back(*calojetc);
    } else if ((calojetc->emEnergyFraction() >= min_EMF_) && (calojetc->n90() >= min_N90_) && (calojetc->emEnergyFraction() <= max_EMF_)) {
      result->push_back(*calojetc);
    }

  } // calojetc

  //iEvent.put( result, "hltJetIDCollection");
  iEvent.put( result);

}
