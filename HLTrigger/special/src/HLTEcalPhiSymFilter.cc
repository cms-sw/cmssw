#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      phiSymEBRecHitCollection->push_back(*itb);
    }
  }

  //Select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    if (ite->energy() >= eCut_endc_) {
      phiSymEERecHitCollection->push_back(*ite);
    }
  }

  if ((!phiSymEBRecHitCollection->size() ) && (!phiSymEBRecHitCollection->size())) 
    return false;

  //Put selected information in the event
  iEvent.put( phiSymEBRecHitCollection, phiSymBarrelHits_);
  iEvent.put( phiSymEERecHitCollection, phiSymEndcapHits_);
  
  iEvent.put(filterproduct);
   
  return true;

}
