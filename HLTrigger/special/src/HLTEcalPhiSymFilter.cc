#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


HLTEcalPhiSymFilter::HLTEcalPhiSymFilter(const edm::ParameterSet& iConfig)
{
  barrelDigis_ = iConfig.getParameter<edm::InputTag> ("barrelDigiCollection");
  endcapDigis_ = iConfig.getParameter<edm::InputTag> ("endcapDigiCollection");
  barrelUncalibHits_ = iConfig.getParameter<edm::InputTag> ("barrelUncalibHitCollection");
  endcapUncalibHits_ = iConfig.getParameter<edm::InputTag> ("endcapUncalibHitCollection");
  barrelHits_ = iConfig.getParameter<edm::InputTag> ("barrelHitCollection");
  endcapHits_ = iConfig.getParameter<edm::InputTag> ("endcapHitCollection");
  phiSymBarrelDigis_ = 
    iConfig.getParameter<std::string> ("phiSymBarrelDigiCollection");
  phiSymEndcapDigis_ = 
    iConfig.getParameter<std::string> ("phiSymEndcapDigiCollection");
  ampCut_barl_ = iConfig.getParameter<double> ("ampCut_barrel");
  ampCut_endc_ = iConfig.getParameter<double> ("ampCut_endcap");
  
  statusThreshold_ = iConfig.getParameter<uint32_t> ("statusThreshold");
  useRecoFlag_ =  iConfig.getParameter<bool>("useRecoFlag");

  barrelDigisToken_ = consumes<EBDigiCollection>(barrelDigis_);
  endcapDigisToken_ = consumes<EEDigiCollection>(endcapDigis_);
  barrelUncalibHitsToken_ = consumes<EcalUncalibratedRecHitCollection>(barrelUncalibHits_);
  endcapUncalibHitsToken_ = consumes<EcalUncalibratedRecHitCollection>(endcapUncalibHits_);
  barrelHitsToken_ = consumes<EBRecHitCollection>(barrelHits_);
  endcapHitsToken_ = consumes<EERecHitCollection>(endcapHits_);

  //register your products
  produces<EBDigiCollection>(phiSymBarrelDigis_);
  produces<EEDigiCollection>(phiSymEndcapDigis_);

}


HLTEcalPhiSymFilter::~HLTEcalPhiSymFilter()
{}

void
HLTEcalPhiSymFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("barrelDigiCollection",edm::InputTag("ecalDigis","ebDigis"));
  desc.add<edm::InputTag>("endcapDigiCollection",edm::InputTag("ecalDigis","eeDigis"));
  desc.add<edm::InputTag>("barrelUncalibHitCollection",edm::InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"));
  desc.add<edm::InputTag>("endcapUncalibHitCollection",edm::InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"));
  desc.add<edm::InputTag>("barrelHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  desc.add<unsigned int>("statusThreshold",3);
  desc.add<bool>("useRecoFlag",false);
  desc.add<double>("ampCut_barrel",8.);
  desc.add<double>("ampCut_endcap",12.);
  desc.add<std::string>("phiSymBarrelDigiCollection","phiSymEcalDigisEB");
  desc.add<std::string>("phiSymEndcapDigiCollection","phiSymEcalDigisEE");
  descriptions.add("alCaPhiSymStream",desc);
}


// ------------ method called to produce the data  ------------
bool
HLTEcalPhiSymFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  using namespace std;
  
  //Get ChannelStatus from DB
  edm::ESHandle<EcalChannelStatus> csHandle;
  if (! useRecoFlag_) iSetup.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus& channelStatus = *csHandle; 

  Handle<EBDigiCollection> barrelDigisHandle;
  Handle<EEDigiCollection> endcapDigisHandle;
  Handle<EcalUncalibratedRecHitCollection> barrelUncalibRecHitsHandle;
  Handle<EcalUncalibratedRecHitCollection> endcapUncalibRecHitsHandle;
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  iEvent.getByToken(barrelDigisToken_,barrelDigisHandle);
  iEvent.getByToken(endcapDigisToken_,endcapDigisHandle);  
  iEvent.getByToken(barrelUncalibHitsToken_,barrelUncalibRecHitsHandle);
  iEvent.getByToken(endcapUncalibHitsToken_,endcapUncalibRecHitsHandle);
  iEvent.getByToken(barrelHitsToken_,barrelRecHitsHandle);
  iEvent.getByToken(endcapHitsToken_,endcapRecHitsHandle);
 
  //Create empty output collections
  std::auto_ptr< EBDigiCollection > phiSymEBDigiCollection( new EBDigiCollection );
  std::auto_ptr< EEDigiCollection > phiSymEEDigiCollection( new EEDigiCollection );
  
  const EBDigiCollection* EBDigis = barrelDigisHandle.product();
  const EEDigiCollection* EEDigis = endcapDigisHandle.product();
  const EBRecHitCollection* EBRechits = barrelRecHitsHandle.product();
  const EERecHitCollection* EERechits = endcapRecHitsHandle.product();

  //Select interesting EcalDigis (barrel)
  EcalUncalibratedRecHitCollection::const_iterator itunb;
  for (itunb=barrelUncalibRecHitsHandle->begin(); itunb!=barrelUncalibRecHitsHandle->end(); itunb++) {
    EcalUncalibratedRecHit hit = (*itunb);
    uint16_t statusCode = 0; 
    if (useRecoFlag_) statusCode=(*EBRechits->find(hit.id())).recoFlag();
    else statusCode = channelStatus[itunb->id().rawId()].getStatusCode();
    float amplitude = hit.amplitude();
    if( statusCode <=statusThreshold_ && amplitude > ampCut_barl_){
        phiSymEBDigiCollection->push_back((*EBDigis->find(hit.id())).id(),(*EBDigis->find(hit.id())).begin());
    }
  }
  
  //Select interesting EcalDigis (endcaps)
  EcalUncalibratedRecHitCollection::const_iterator itune;
  for (itune=endcapUncalibRecHitsHandle->begin(); itune!=endcapUncalibRecHitsHandle->end(); itune++) {
    EcalUncalibratedRecHit hit = (*itune);
    uint16_t statusCode = 0; 
    if (useRecoFlag_) statusCode=(*EERechits->find(hit.id())).recoFlag();
    else statusCode = channelStatus[itune->id().rawId()].getStatusCode();
    float amplitude = hit.amplitude();
    if( statusCode <=statusThreshold_ && amplitude > ampCut_endc_){
        phiSymEEDigiCollection->push_back((*EEDigis->find(hit.id())).id(),(*EEDigis->find(hit.id())).begin());
    }
  }

  if ((!phiSymEBDigiCollection->size() ) && (!phiSymEEDigiCollection->size())) 
    return false;

  //Put selected information in the event
  iEvent.put( phiSymEBDigiCollection, phiSymBarrelDigis_);
  iEvent.put( phiSymEEDigiCollection, phiSymEndcapDigis_);
  
  return true;
}
