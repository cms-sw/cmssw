/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecHitProducer.cc,v 1.1 2006/03/10 08:43:16 rahatlou Exp $
 *  $Date: 2006/03/10 08:43:16 $
 *  $Revision: 1.1 $
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

#include <iostream>
#include <cmath>
#include <vector>

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


EcalRecHitProducer::EcalRecHitProducer(const edm::ParameterSet& ps) {

   uncalibRecHitCollection_ = ps.getParameter<std::string>("uncalibRecHitCollection");
   uncalibRecHitProducer_   = ps.getParameter<std::string>("uncalibRecHitProducer");
   rechitCollection_        = ps.getParameter<std::string>("rechitCollection");
   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

   algo_ = new EcalRecHitSimpleAlgo();

   produces< EcalRecHitCollection >(rechitCollection_);
   nEvt_ = 0; // reset local event counter
}

EcalRecHitProducer::~EcalRecHitProducer() {

  delete algo_;

}

void
EcalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   nEvt_++;

   Handle< EcalUncalibratedRecHitCollection > pUncalibRecHits;
   try {
     evt.getByLabel( uncalibRecHitProducer_, uncalibRecHitCollection_, pUncalibRecHits);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product " << uncalibRecHitCollection_.c_str() << std::endl;
   }
   const EcalUncalibratedRecHitCollection*  uncalibRecHits = pUncalibRecHits.product(); // get a ptr to the product
   if(!counterExceeded()) {
      std::cout << "EcalRecHitProducer: total # uncalibrated rechits: " << uncalibRecHits->size()
                << std::endl;
   }

   // now fetch all conditions we need to make rechits
   // ADC -> GeV Scale
   edm::ESHandle<EcalADCToGeVConstant> pAgc;
   es.get<EcalADCToGeVConstantRcd>().get(pAgc);
   const EcalADCToGeVConstant* agc = pAgc.product();
   //std::cout << "Global ADC->GeV scale: " << agc->getValue() << " GeV/ADC count" << std::endl;
   //
   // use this value in the algorithm
   algo_->setADCToGeVConstant(float(agc->getValue()));

   // Intercalib constants
   edm::ESHandle<EcalIntercalibConstants> pIcal;
   es.get<EcalIntercalibConstantsRcd>().get(pIcal);
   const EcalIntercalibConstants* ical = pIcal.product();


   // collection of rechits to put in the event
   std::auto_ptr< EcalRecHitCollection > rechits( new EcalRecHitCollection );

   // loop over uncalibrated rechits to make calibrated ones
   for(EcalUncalibratedRecHitCollection::const_iterator it  = uncalibRecHits->begin();
                                                        it != uncalibRecHits->end(); ++it) {

     // find intercalib constant for this xtal
     EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(it->id().rawId());
     EcalIntercalibConstants::EcalIntercalibConstant icalconst;
     if( icalit!=ical->getMap().end() ){
       icalconst = icalit->second;
     } else {
      std::cout << "No intercalib const found for this xtal! something wrong with EcalIntercalibConstants in your DB? "
                << std::endl;
     }

     // make the rechit and put in the output collection
     // must implement op= for EcalRecHit
     EcalRecHit aHit( algo_->makeRecHit(*it, icalconst ) );
     rechits->push_back( aHit );

  /**
     if(it->amplitude()>0. && !counterExceeded() ) {
        std::cout << "EcalRecHitProducer: processed UncalibRecHit with rawId: "
                  << it->id().rawId() << "\n"
                  << "uncalib rechit amplitude: " << it->amplitude()
                  << " calib rechit energy: " << aHit.energy()
                  << std::endl;
     }
  **/
   }

   // put the collection of recunstructed hits in the event

   if(!counterExceeded()) {
      std::cout << "EcalRecHitProducer: total # calibrated rechits: " << rechits->size()
                << std::endl;
   }

   evt.put( rechits, rechitCollection_ );
} //produce()

