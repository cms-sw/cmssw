
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitMiscalib.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRecHitMiscalib::EcalRecHitMiscalib(const edm::ParameterSet& iConfig)
{
  ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");
  barrelHits_ = iConfig.getParameter< std::string > ("barrelHitCollection");
  endcapHits_ = iConfig.getParameter< std::string > ("endcapHitCollection");
  RecalibBarrelHits_ = iConfig.getParameter< std::string > ("RecalibBarrelHitCollection");
  RecalibEndcapHits_ = iConfig.getParameter< std::string > ("RecalibEndcapHitCollection");

  //register your products
  produces< EBRecHitCollection >(RecalibBarrelHits_);
  produces< EERecHitCollection >(RecalibEndcapHits_);
}


EcalRecHitMiscalib::~EcalRecHitMiscalib()
{
 

}


// ------------ method called to produce the data  ------------
void
EcalRecHitMiscalib::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  const EBRecHitCollection*  EBRecHits = 0;
  const EERecHitCollection*  EERecHits = 0; 
 
  try {
    iEvent.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
    EBRecHits = barrelRecHitsHandle.product(); // get a ptr to the product

    iEvent.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
    EERecHits = endcapRecHitsHandle.product(); // get a ptr to the product
  } catch ( std::exception& ex ) {
    LogDebug("") << "EcalREcHitMiscalib: Error! can't get product!" << std::endl;
  }

  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > RecalibEBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EERecHitCollection > RecalibEERecHitCollection( new EERecHitCollection );


  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();

  if(EBRecHits)
    {

       //loop on all EcalRecHits (barrel)
      EBRecHitCollection::const_iterator itb;
      for (itb=EBRecHits->begin(); itb!=EBRecHits->end(); itb++) {
	
	// find intercalib constant for this xtal
	EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(itb->id().rawId());
	EcalIntercalibConstants::EcalIntercalibConstant icalconst;

	if( icalit!=ical->getMap().end() ){
	  icalconst = icalit->second;
	  // edm::LogDebug("EcalRecHitMiscalib") << "Found intercalib for xtal " << EBDetId(itb->id()) << " " << icalconst ;

	} else {
	  edm::LogError("EcalRecHitMiscalib") << "No intercalib const found for xtal " << EBDetId(itb->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
              ;
          }
          
          // make the rechit with rescaled energy and put in the output collection

	  EcalRecHit aHit(itb->id(),itb->energy()*icalconst,itb->time());
	  
	  RecalibEBRecHitCollection->push_back( aHit);
      }
    }

  if(EERecHits)
    {

       //loop on all EcalRecHits (barrel)
      EERecHitCollection::const_iterator ite;
      for (ite=EERecHits->begin(); ite!=EERecHits->end(); ite++) {
	
	// find intercalib constant for this xtal
	EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(ite->id().rawId());
	EcalIntercalibConstants::EcalIntercalibConstant icalconst;

	if( icalit!=ical->getMap().end() ){
	  icalconst = icalit->second;
	  // edm:: LogDebug("EcalRecHitMiscalib") << "Found intercalib for xtal " << EEDetId(ite->id()) << " " << icalconst ;
          } else {
            edm::LogError("EcalRecHitMiscalib") << "No intercalib const found for xtal " << EEDetId(ite->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
              ;
          }
          
          // make the rechit with rescaled energy and put in the output collection

	  EcalRecHit aHit(ite->id(),ite->energy()*icalconst,ite->time());
	  
	  RecalibEERecHitCollection->push_back( aHit);
      }
    }


  //Put Recalibrated rechit in the event
  iEvent.put( RecalibEBRecHitCollection, RecalibBarrelHits_);
  iEvent.put( RecalibEERecHitCollection, RecalibEndcapHits_);
  
}

//define it as module
//DEFINE_ANOTHER_FWK_MODULE(EcalRecHitMiscalib);
