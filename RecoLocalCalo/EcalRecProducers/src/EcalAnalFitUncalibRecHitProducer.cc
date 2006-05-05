/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytical fit method
 *
  *  $Id: EcalAnalFitUncalibRecHitProducer.cc,v 1.4 2006/04/07 12:47:07 meridian Exp $
  *  $Date: 2006/04/07 12:47:07 $
  *  $Revision: 1.4 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalAnalFitUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "CLHEP/Matrix/Matrix.h"
//#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

EcalAnalFitUncalibRecHitProducer::EcalAnalFitUncalibRecHitProducer(const edm::ParameterSet& ps) {

   EBdigiCollection_ = ps.getParameter<std::string>("EBdigiCollection");
   EEdigiCollection_ = ps.getParameter<std::string>("EEdigiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
   EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");
   //   nMaxPrintout_   = ps.getUntrackedParameter<int>("nMaxPrintout",10);
   produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
   produces< EEUncalibratedRecHitCollection >(EEhitCollection_);
   //   nEvt_ = 0; // reset local event counter
}

EcalAnalFitUncalibRecHitProducer::~EcalAnalFitUncalibRecHitProducer() {
}

void
EcalAnalFitUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   //   nEvt_++;

   Handle< EBDigiCollection > pEBDigis;
   Handle< EEDigiCollection > pEEDigis;

   const EBDigiCollection* EBdigis =0;
   const EEDigiCollection* EEdigis =0;
   
   try {
     //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
     evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the produc
     edm::LogInfo("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: total # EBdigis: " << EBdigis->size() ;
   } catch ( std::exception& ex ) {
     //     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
   }

   try {
     //     evt.getByLabel( digiProducer_, EEdigiCollection_, pEEDigis);
     evt.getByLabel( digiProducer_, pEEDigis);
     EEdigis = pEEDigis.product(); // get a ptr to the product
     edm::LogInfo("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: total # EEdigis: " << EEdigis->size() ;
   } catch ( std::exception& ex ) {
     //     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_.c_str() ;
   }



//    if(!counterExceeded()) 
//      {


//      }


   // collection of reco'ed ampltudes to put in the event

   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

   std::vector<double> pedVec;
   std::vector<HepMatrix> weights;
   std::vector<HepSymMatrix> chi2mat;

   if (EBdigis)
     {
       for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {
	 
	 EcalUncalibratedRecHit aHit =
	   EBalgo_.makeRecHit(*itdg, pedVec, weights, chi2mat);
	 EBuncalibRechits->push_back( aHit );
	 
	 if(aHit.amplitude()>0.) {
	   LogDebug("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: processed EBDataFrame with id: "
					     << itdg->id() << "\n"
					     << "uncalib rechit amplitude: " << aHit.amplitude()
	     ;
	 }
	 
       }
     }

   // put the collection of recunstructed hits in the event
   evt.put( EBuncalibRechits, EBhitCollection_ );
   evt.put( EEuncalibRechits, EEhitCollection_ );
}

