#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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

  //register your products
  produces< EBRecHitCollection >(phiSymBarrelHits_);
  produces< EERecHitCollection >(phiSymEndcapHits_);
  produces<trigger::TriggerFilterObjectWithRefs>();


}


HLTEcalPhiSymFilter::~HLTEcalPhiSymFilter()
{
 

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

  
  iEvent.getByLabel(barrelHits_,barrelRecHitsHandle);
  iEvent.getByLabel(endcapHits_,endcapRecHitsHandle);
 
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > phiSymEBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EERecHitCollection > phiSymEERecHitCollection( new EERecHitCollection );

  // The Filter object. We don't really need to put anything into it, but we 
  // write an empty one for consistency
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> 
      filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  

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
  
  iEvent.put(filterproduct);
   
  return true;

}
