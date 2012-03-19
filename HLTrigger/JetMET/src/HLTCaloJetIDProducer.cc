#include "HLTrigger/JetMET/interface/HLTCaloJetIDProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

HLTCaloJetIDProducer::HLTCaloJetIDProducer(const edm::ParameterSet& iConfig) :
  jetsInput_   (iConfig.getParameter<edm::InputTag>("jetsInput")),
  min_EMF_     (iConfig.getParameter<double>("min_EMF")),
  max_EMF_     (iConfig.getParameter<double>("max_EMF")),
  min_N90_     (iConfig.getParameter<int>("min_N90")),
  min_N90hits_ (iConfig.getParameter<int>("min_N90hits")),
  jetID_       (iConfig.getParameter<edm::ParameterSet>("JetIDParams"))
{
  //  produces< reco::CaloJetCollection > ( "hltCaloJetIDCollection" );
  produces< reco::CaloJetCollection > ();
}

void HLTCaloJetIDProducer::beginJob()
{

}

HLTCaloJetIDProducer::~HLTCaloJetIDProducer()
{

}

void HLTCaloJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::CaloJetCollection> calojets;
  iEvent.getByLabel(jetsInput_, calojets);

  std::auto_ptr<reco::CaloJetCollection> result (new reco::CaloJetCollection);

  for (reco::CaloJetCollection::const_iterator calojetc = calojets->begin(); 
       calojetc != calojets->end(); ++calojetc) {
      
    if (std::abs(calojetc->eta()) >= 2.6) {
      result->push_back(*calojetc);
    } else {
      if (min_N90hits_>0) jetID_.calculate( iEvent, *calojetc );
      if ((calojetc->emEnergyFraction() >= min_EMF_) && ((min_N90hits_<=0) || (jetID_.n90Hits() >= min_N90hits_))  && (calojetc->n90() >= min_N90_) && (calojetc->emEnergyFraction() <= max_EMF_)) {
	result->push_back(*calojetc);
      }
    }
  } // calojetc

  //iEvent.put( result, "hltCaloJetIDCollection");
  iEvent.put( result);

}
