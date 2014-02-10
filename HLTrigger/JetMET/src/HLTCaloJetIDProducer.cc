#include "HLTrigger/JetMET/interface/HLTCaloJetIDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h" 

HLTCaloJetIDProducer::HLTCaloJetIDProducer(const edm::ParameterSet& iConfig) :
  jetsInput_   (iConfig.getParameter<edm::InputTag>("jetsInput")),
  min_EMF_     (iConfig.getParameter<double>("min_EMF")),
  max_EMF_     (iConfig.getParameter<double>("max_EMF")),
  min_N90_     (iConfig.getParameter<int>("min_N90")),
  min_N90hits_ (iConfig.getParameter<int>("min_N90hits")),
  jetID_       (iConfig.getParameter<edm::ParameterSet>("JetIDParams"), consumesCollector())
{
  //  produces< reco::CaloJetCollection > ( "hltCaloJetIDCollection" );
  produces< reco::CaloJetCollection > ();
  m_theCaloJetToken = consumes<reco::CaloJetCollection>(jetsInput_);
}

void HLTCaloJetIDProducer::beginJob()
{

}

HLTCaloJetIDProducer::~HLTCaloJetIDProducer()
{

}

void 
HLTCaloJetIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<double>("min_EMF",0.0001);
  desc.add<double>("max_EMF",999.);
  desc.add<int>("min_N90",0);
  desc.add<int>("min_N90hits",2);
  edm::ParameterSetDescription jetidPSet;
  jetidPSet.add<bool>("useRecHits",true);
  jetidPSet.add<edm::InputTag>("hbheRecHitsColl",edm::InputTag("hltHbhereco"));
  jetidPSet.add<edm::InputTag>("hoRecHitsColl",edm::InputTag("hltHoreco"));
  jetidPSet.add<edm::InputTag>("hfRecHitsColl",edm::InputTag("hltHfreco"));
  jetidPSet.add<edm::InputTag>("ebRecHitsColl",edm::InputTag("hltEcalRecHitAll", "EcalRecHitsEB"));
  jetidPSet.add<edm::InputTag>("eeRecHitsColl",edm::InputTag("hltEcalRecHitAll", "EcalRecHitsEE"));
  desc.add<edm::ParameterSetDescription>("JetIDParams",jetidPSet);
  descriptions.add("hltCaloJetIDProducer",desc);
}


void HLTCaloJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::CaloJetCollection> calojets;
  iEvent.getByToken(m_theCaloJetToken, calojets);

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
