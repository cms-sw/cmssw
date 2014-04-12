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
  barrelHits_ = iConfig.getParameter< edm::InputTag > ("barrelHitCollection");
  endcapHits_ = iConfig.getParameter< edm::InputTag > ("endcapHitCollection");
  phiSymBarrelHits_ = 
    iConfig.getParameter< std::string > ("phiSymBarrelHitCollection");
  phiSymEndcapHits_ = 
    iConfig.getParameter< std::string > ("phiSymEndcapHitCollection");
  eCut_barl_ = iConfig.getParameter< double > ("eCut_barrel");
  eCut_endc_ = iConfig.getParameter< double > ("eCut_endcap");
  eCut_barl_high_=iConfig.getParameter< double > ("eCut_barrel_high");
  eCut_endc_high_=iConfig.getParameter< double > ("eCut_endcap_high");

  statusThreshold_ = iConfig.getParameter<uint32_t> ("statusThreshold");
  useRecoFlag_ =  iConfig.getParameter<bool>("useRecoFlag");

  barrelHitsToken_ = consumes<EBRecHitCollection>(barrelHits_);
  endcapHitsToken_ = consumes<EERecHitCollection>(endcapHits_);

  //register your products
  produces< EBRecHitCollection >(phiSymBarrelHits_);
  produces< EERecHitCollection >(phiSymEndcapHits_);
}


HLTEcalPhiSymFilter::~HLTEcalPhiSymFilter()
{}

void
HLTEcalPhiSymFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("barrelHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  desc.add<unsigned int>("statusThreshold",3);
  desc.add<bool>("useRecoFlag",false);
  desc.add<double>("eCut_barrel",150.);
  desc.add<double>("eCut_endcap",750.);
  desc.add<double>("eCut_barrel_high",999999.);
  desc.add<double>("eCut_endcap_high",999999.);
  desc.add<std::string>("phiSymBarrelHitCollection","phiSymEcalRecHitsEB");
  desc.add<std::string>("phiSymEndcapHitCollection","phiSymEcalRecHitsEE");
  descriptions.add("alCaPhiSymStream",desc);
}


// ------------ method called to produce the data  ------------
bool
HLTEcalPhiSymFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  using namespace edm;
  using namespace std;
  
  edm::ESHandle<EcalChannelStatus> csHandle;
  if (! useRecoFlag_) iSetup.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus& channelStatus = *csHandle; 
  


  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  
  iEvent.getByToken(barrelHitsToken_,barrelRecHitsHandle);
  iEvent.getByToken(endcapHitsToken_,endcapRecHitsHandle);
 
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > phiSymEBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EERecHitCollection > phiSymEERecHitCollection( new EERecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    if (itb->energy() >= eCut_barl_) {
      uint16_t statusCode = 0; 
      if (useRecoFlag_) statusCode=itb->recoFlag();
      else statusCode = channelStatus[itb->id().rawId()].getStatusCode();
      if ( statusCode <=statusThreshold_ ) 
	phiSymEBRecHitCollection->push_back(*itb);
      else if  (itb->energy() >= eCut_barl_high_ ) 
	phiSymEBRecHitCollection->push_back(*itb);
    }
  }
  
  //Select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    if (ite->energy() >= eCut_endc_) {
       uint16_t statusCode = 0; 
       if (useRecoFlag_) statusCode=ite->recoFlag();
       else statusCode =channelStatus[ite->id().rawId()].getStatusCode(); 
       if ( statusCode <=statusThreshold_ ) 
	 phiSymEERecHitCollection->push_back(*ite);
       else if  (ite->energy() >= eCut_endc_high_ ) 
	 phiSymEERecHitCollection->push_back(*ite);
    }
  }

  if ((!phiSymEBRecHitCollection->size() ) && (!phiSymEERecHitCollection->size())) 
    return false;
  
  //Put selected information in the event
  iEvent.put( phiSymEBRecHitCollection, phiSymBarrelHits_);
  iEvent.put( phiSymEERecHitCollection, phiSymEndcapHits_);
  
  return true;
}
