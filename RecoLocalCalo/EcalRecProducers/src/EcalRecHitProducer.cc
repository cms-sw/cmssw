/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecHitProducer.cc,v 1.3 2006/04/07 12:47:07 meridian Exp $
 *  $Date: 2006/04/07 12:47:07 $
 *  $Revision: 1.3 $
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>
#include <vector>

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"



EcalRecHitProducer::EcalRecHitProducer(const edm::ParameterSet& ps) {

   EBuncalibRecHitCollection_ = ps.getParameter<std::string>("EBuncalibRecHitCollection");
   EEuncalibRecHitCollection_ = ps.getParameter<std::string>("EEuncalibRecHitCollection");
   uncalibRecHitProducer_   = ps.getParameter<std::string>("uncalibRecHitProducer");
   EBrechitCollection_        = ps.getParameter<std::string>("EBrechitCollection");
   EErechitCollection_        = ps.getParameter<std::string>("EErechitCollection");
   //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

   EBalgo_ = new EcalRecHitSimpleAlgo();
   EEalgo_ = new EcalRecHitSimpleAlgo();

   produces< EBRecHitCollection >(EBrechitCollection_);
   produces< EERecHitCollection >(EErechitCollection_);

   //   nEvt_ = 0; // reset local event counter
}

EcalRecHitProducer::~EcalRecHitProducer() {

  if (EBalgo_) delete EBalgo_;
  if (EEalgo_) delete EEalgo_;

}

void
EcalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   //   nEvt_++;

   Handle< EBUncalibratedRecHitCollection > pEBUncalibRecHits;
   Handle< EEUncalibratedRecHitCollection > pEEUncalibRecHits;

   const EBUncalibratedRecHitCollection*  EBuncalibRecHits = 0;
   const EEUncalibratedRecHitCollection*  EEuncalibRecHits = 0; 

   try {
     evt.getByLabel( uncalibRecHitProducer_, EBuncalibRecHitCollection_, pEBUncalibRecHits);
     EBuncalibRecHits = pEBUncalibRecHits.product(); // get a ptr to the product
     LogDebug("EcalRecHitDebug") << "total # EB uncalibrated rechits: " << EBuncalibRecHits->size();
   } catch ( std::exception& ex ) {
     //edm::LogError("EcalRecHitError") << "Error! can't get the product " << EBuncalibRecHitCollection_.c_str() ;
   }
   
   try {
     evt.getByLabel( uncalibRecHitProducer_, EEuncalibRecHitCollection_, pEEUncalibRecHits);
     EEuncalibRecHits = pEEUncalibRecHits.product(); // get a ptr to the product
     LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits: " << EEuncalibRecHits->size();
   } catch ( std::exception& ex ) {
     //edm::LogError("EcalRecHitError") << "Error! can't get the product " << EEuncalibRecHitCollection_.c_str() ;
   }

   // collection of rechits to put in the event
   std::auto_ptr< EBRecHitCollection > EBrechits( new EBRecHitCollection );
   std::auto_ptr< EERecHitCollection > EErechits( new EERecHitCollection );

   // now fetch all conditions we nEEd to make rechits
   // ADC -> GeV Scale
   // TODO Make two ADCtoGeV scale for EB & EE 

   if (EBuncalibRecHits)
     {
       edm::ESHandle<EcalADCToGeVConstant> pAgc;
       es.get<EcalADCToGeVConstantRcd>().get(pAgc);
       const EcalADCToGeVConstant* agc = pAgc.product();
       LogDebug("EcalRecHitDebug") << "Global ADC->GeV scale: " << agc->getValue() << " GeV/ADC count" ;
       //
       // use this value in the algorithm
       EBalgo_->setADCToGeVConstant(float(agc->getValue()));
       
       // Intercalib constants
       edm::ESHandle<EcalIntercalibConstants> pIcal;
       es.get<EcalIntercalibConstantsRcd>().get(pIcal);
       const EcalIntercalibConstants* ical = pIcal.product();
       
       // loop over uncalibrated rechits to make calibrated ones
       for(EBUncalibratedRecHitCollection::const_iterator it  = EBuncalibRecHits->begin();
	   it != EBuncalibRecHits->end(); ++it) {
	 
	 // find intercalib constant for this xtal
	 EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(it->id().rawId());
	 EcalIntercalibConstants::EcalIntercalibConstant icalconst;
	 if( icalit!=ical->getMap().end() ){
	   icalconst = icalit->second;
	   LogDebug("EcalRecHitDebug") << "Found intercalib for xtal " << EBDetId(it->id()).ic() << " " << icalconst ;
	 } else {
	   edm::LogError("EcalRecHitDebug") << "No intercalib const found for xtal " << EBDetId(it->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	     ;
	 }
	 
	 // make the rechit and put in the output collection
	 // must implement op= for EcalRecHit
	 EcalRecHit aHit( EBalgo_->makeRecHit(*it, icalconst ) );
	 EBrechits->push_back( aHit );
	 
	 
	 if(it->amplitude()>0.) 
	   {
	     edm::LogInfo("EcalRecHitDebug") << "processed UncalibRecHit with rawId: "
					     << it->id().rawId() << "\n"
					     << "uncalib rechit amplitude: " << it->amplitude()
					     << " calib rechit energy: " << aHit.energy()
	       ;
	   }
       }
     }
   // put the collection of recunstructed hits in the event
   
   edm::LogInfo("EcalRecHitInfo") << "total # EB calibrated rechits: " << EBrechits->size()
     ;
   edm::LogInfo("EcalRecHitInfo") << "total # EE calibrated rechits: " << EErechits->size()
     ;

   evt.put( EBrechits, EBrechitCollection_ );
   evt.put( EErechits, EErechitCollection_ );
} //produce()

