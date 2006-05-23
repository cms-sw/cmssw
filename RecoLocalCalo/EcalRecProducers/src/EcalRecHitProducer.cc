/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecHitProducer.cc,v 1.5 2006/04/21 10:45:54 meridian Exp $
 *  $Date: 2006/04/21 10:45:54 $
 *  $Revision: 1.5 $
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
#include "DataFormats/EcalDetId/interface/EEDetId.h"
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
       LogDebug("EcalRecHitDebug") << "Global EB ADC->GeV scale: " << agc->getEBValue() << " GeV/ADC count" ;
       //
       // use this value in the algorithm
       EBalgo_->setADCToGeVConstant(float(agc->getEBValue()));
       
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
	   //	   LogDebug("EcalRecHitDebug") << "Found intercalib for xtal " << EBDetId(it->id()).ic() << " " << icalconst ;
	 } else {
	   edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EBDetId(it->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	     ;
	 }
	 
	 // make the rechit and put in the output collection
	 // must implement op= for EcalRecHit
	 EcalRecHit aHit( EBalgo_->makeRecHit(*it, icalconst ) );
	 EBrechits->push_back( aHit );
	 
	 
	 if(it->amplitude()>0.) 
	   {
	     LogDebug("EcalRecHitDebug") << "processed UncalibRecHit with rawId: "
					     << it->id().rawId() << "\n"
					     << "uncalib rechit amplitude: " << it->amplitude()
					     << " calib rechit energy: " << aHit.energy()
	       ;
	   }
       }
     }

   if (EEuncalibRecHits)
     {
       edm::ESHandle<EcalADCToGeVConstant> pAgc;
       es.get<EcalADCToGeVConstantRcd>().get(pAgc);
       const EcalADCToGeVConstant* agc = pAgc.product();
       LogDebug("EcalRecHitDebug") << "Global ADC->GeV scale: " << agc->getEEValue() << " GeV/ADC count" ;
       //
       // use this value in the algorithm
       EEalgo_->setADCToGeVConstant(float(agc->getEEValue()));
       
       // Intercalib constants
       edm::ESHandle<EcalIntercalibConstants> pIcal;
       es.get<EcalIntercalibConstantsRcd>().get(pIcal);
       const EcalIntercalibConstants* ical = pIcal.product();
       
       // loop over uncalibrated rechits to make calibrated ones
       for(EEUncalibratedRecHitCollection::const_iterator it  = EEuncalibRecHits->begin();
	   it != EEuncalibRecHits->end(); ++it) {
	 
	 // find intercalib constant for this xtal
	 EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(it->id().rawId());
	 EcalIntercalibConstants::EcalIntercalibConstant icalconst;
	 if( icalit!=ical->getMap().end() ){
	   icalconst = icalit->second;
	   // LogDebug("EcalRecHitDebug") << "Found intercalib for xtal " << EEDetId(it->id()).ic() << " " << icalconst ;
	 } else {
	   edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EEDetId(it->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
	     ;
	 }
	 
	 // make the rechit and put in the output collection
	 // must implement op= for EcalRecHit
	 EcalRecHit aHit( EEalgo_->makeRecHit(*it, icalconst ) );
	 EErechits->push_back( aHit );
	 
	 
	 if(it->amplitude()>0.) 
	   {
	     LogDebug("EcalRecHitDebug") << "processed UncalibRecHit with rawId: "
					     << it->id().rawId() << "\n"
					     << "uncalib rechit amplitude: " << it->amplitude()
					     << " calib rechit energy: " << aHit.energy()
	       ;
	   }
       }
     }
   // put the collection of recunstructed hits in the event
   
   LogDebug("EcalRecHitInfo") << "total # EB calibrated rechits: " << EBrechits->size()
     ;
   LogDebug("EcalRecHitInfo") << "total # EE calibrated rechits: " << EErechits->size()
     ;

   evt.put( EBrechits, EBrechitCollection_ );
   evt.put( EErechits, EErechitCollection_ );
} //produce()

