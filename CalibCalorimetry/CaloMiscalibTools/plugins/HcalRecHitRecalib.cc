
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/HcalRecHitRecalib.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLHcal.h"

HcalRecHitRecalib::HcalRecHitRecalib(const edm::ParameterSet& iConfig)
{

  hbheLabel_ = iConfig.getParameter<edm::InputTag>("hbheInput");
  hoLabel_ = iConfig.getParameter<edm::InputTag>("hoInput");
  hfLabel_ = iConfig.getParameter<edm::InputTag>("hfInput");



//   HBHEHitsProducer_ = iConfig.getParameter< std::string > ("HBHERecHitsProducer");
//   HOHitsProducer_ = iConfig.getParameter< std::string > ("HERecHitsProducer");
//   HFHitsProducer_ = iConfig.getParameter< std::string > ("HERecHitsProducer");
//   HBHEHits_ = iConfig.getParameter< std::string > ("HBHEHitCollection");
//   HFHits_ = iConfig.getParameter< std::string > ("HFHitCollection");
//   HOHits_ = iConfig.getParameter< std::string > ("HOHitCollection");

  RecalibHBHEHits_ = iConfig.getParameter< std::string > ("RecalibHBHEHitCollection");
  RecalibHFHits_ = iConfig.getParameter< std::string > ("RecalibHFHitCollection");
  RecalibHOHits_ = iConfig.getParameter< std::string > ("RecalibHOHitCollection");

  //register your products
  produces< HBHERecHitCollection >(RecalibHBHEHits_);
  produces< HFRecHitCollection >(RecalibHFHits_);
  produces< HORecHitCollection >(RecalibHOHits_);

  // here read them from xml (particular to HCAL)
  mapHcal_.prefillMap();
  hcalfile_=iConfig.getUntrackedParameter<std::string> ("fileNameHcal","");
  MiscalibReaderFromXMLHcal hcalreader_(mapHcal_);
  if(!hcalfile_.empty()) hcalreader_.parseXMLMiscalibFile(hcalfile_);
  mapHcal_.print();

}


HcalRecHitRecalib::~HcalRecHitRecalib()
{
 

}


// ------------ method called to produce the data  ------------
void
HcalRecHitRecalib::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  Handle<HBHERecHitCollection> HBHERecHitsHandle;
  Handle<HFRecHitCollection> HFRecHitsHandle;
  Handle<HORecHitCollection> HORecHitsHandle;

  const HBHERecHitCollection*  HBHERecHits = 0;
  const HFRecHitCollection*  HFRecHits = 0;
  const HORecHitCollection*  HORecHits = 0;

 try {
    iEvent.getByLabel(hbheLabel_,HBHERecHitsHandle);
    HBHERecHits = HBHERecHitsHandle.product(); // get a ptr to the product

    iEvent.getByLabel(hoLabel_,HORecHitsHandle);
    HORecHits = HORecHitsHandle.product(); // get a ptr to the product

    iEvent.getByLabel(hfLabel_,HFRecHitsHandle);
    HFRecHits = HFRecHitsHandle.product(); // get a ptr to the product

  } catch ( std::exception& ex ) {
    LogDebug("") << "HcalREcHitRecalib: Error! can't get product!" << std::endl;
  }



//     iEvent.getByLabel(HBHEHitsProducer_,HBHEHits_,HBHERecHitsHandle);
//     HBHERecHits = HBHERecHitsHandle.product(); // get a ptr to the product

//     iEvent.getByLabel(HFHitsProducer_,HFHits_,HFRecHitsHandle);
//     HFRecHits = HFRecHitsHandle.product(); // get a ptr to the product

//     iEvent.getByLabel(HOHitsProducer_,HOHits_,HORecHitsHandle);
//     HORecHits = HORecHitsHandle.product(); // get a ptr to the product


  //Create empty output collections
  std::auto_ptr< HBHERecHitCollection > RecalibHBHERecHitCollection( new HBHERecHitCollection );
  std::auto_ptr< HFRecHitCollection > RecalibHFRecHitCollection( new HFRecHitCollection );
  std::auto_ptr< HORecHitCollection > RecalibHORecHitCollection( new HORecHitCollection );

  // Intercalib constants
  //  edm::ESHandle<EcalIntercalibConstants> pIcal;
  //  iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
  //  const EcalIntercalibConstants* ical = pIcal.product();

  if(HBHERecHits)
    {

       //loop on all EcalRecHits (barrel)
      HBHERecHitCollection::const_iterator itHBHE;
      for (itHBHE=HBHERecHits->begin(); itHBHE!=HBHERecHits->end(); itHBHE++) {
	
	// find intercalib constant for this cell

	//	EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(itb->id().rawId());
	//	EcalIntercalibConstants::EcalIntercalibConstant icalconst;

	//	if( icalit!=ical->getMap().end() ){
	//	  icalconst = icalit->second;
	  // edm::LogDebug("EcalRecHitMiscalib") << "Found intercalib for xtal " << EBDetId(itb->id()) << " " << icalconst ;

	//	} else {
	//	  edm::LogError("EcalRecHitMiscalib") << "No intercalib const found for xtal " << EBDetId(itb->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	//              ;
	//          }
          
	float icalconst=(mapHcal_.get().find(itHBHE->id().rawId()))->second;
          // make the rechit with rescaled energy and put in the output collection

	HBHERecHit aHit(itHBHE->id(),itHBHE->energy()*icalconst,itHBHE->time());
	
	RecalibHBHERecHitCollection->push_back( aHit);
      }
    }

  if(HFRecHits)
    {

       //loop on all EcalRecHits (barrel)
      HFRecHitCollection::const_iterator itHF;
      for (itHF=HFRecHits->begin(); itHF!=HFRecHits->end(); itHF++) {
	
	// find intercalib constant for this cell

	//	EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(itb->id().rawId());
	//	EcalIntercalibConstants::EcalIntercalibConstant icalconst;

	//	if( icalit!=ical->getMap().end() ){
	//	  icalconst = icalit->second;
	  // edm::LogDebug("EcalRecHitMiscalib") << "Found intercalib for xtal " << EBDetId(itb->id()) << " " << icalconst ;

	//	} else {
	//	  edm::LogError("EcalRecHitMiscalib") << "No intercalib const found for xtal " << EBDetId(itb->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	//              ;
	//          }
          
          // make the rechit with rescaled energy and put in the output collection
	
	float icalconst=(mapHcal_.get().find(itHF->id().rawId()))->second;
	HFRecHit aHit(itHF->id(),itHF->energy()*icalconst,itHF->time());
	
	RecalibHFRecHitCollection->push_back( aHit);
      }
    }

  if(HORecHits)
    {

       //loop on all EcalRecHits (barrel)
      HORecHitCollection::const_iterator itHO;
      for (itHO=HORecHits->begin(); itHO!=HORecHits->end(); itHO++) {
	
	// find intercalib constant for this cell

	//	EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(itb->id().rawId());
	//	EcalIntercalibConstants::EcalIntercalibConstant icalconst;

	//	if( icalit!=ical->getMap().end() ){
	//	  icalconst = icalit->second;
	  // edm::LogDebug("EcalRecHitMiscalib") << "Found intercalib for xtal " << EBDetId(itb->id()) << " " << icalconst ;

	//	} else {
	//	  edm::LogError("EcalRecHitMiscalib") << "No intercalib const found for xtal " << EBDetId(itb->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	//              ;
	//          }
          
          // make the rechit with rescaled energy and put in the output collection

	float icalconst=(mapHcal_.get().find(itHO->id().rawId()))->second;
	HORecHit aHit(itHO->id(),itHO->energy()*icalconst,itHO->time());
	  
	  RecalibHORecHitCollection->push_back( aHit);
      }
    }


  //Put Recalibrated rechit in the event
  iEvent.put( RecalibHBHERecHitCollection, RecalibHBHEHits_);
  iEvent.put( RecalibHFRecHitCollection, RecalibHFHits_);
  iEvent.put( RecalibHORecHitCollection, RecalibHOHits_);
}

