/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytical fit method
 *
  *  $Id: EcalAnalFitUncalibRecHitProducer.cc,v 1.3 2006/01/10 11:28:43 meridian Exp $
  *  $Date: 2006/01/10 11:28:43 $
  *  $Revision: 1.3 $
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

   try {
     //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
     evt.getByLabel( digiProducer_, pEBDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
   }

   try {
     //     evt.getByLabel( digiProducer_, EEdigiCollection_, pEEDigis);
     evt.getByLabel( digiProducer_, pEEDigis);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_.c_str() ;
   }

   const EBDigiCollection* EBdigis = pEBDigis.product(); // get a ptr to the produc
   const EEDigiCollection* EEdigis = pEEDigis.product(); // get a ptr to the product

//    if(!counterExceeded()) 
//      {
   edm::LogInfo("EcalUncalibRecHitInfo") << "EcalWeightUncalibRecHitProducer: total # EBdigis: " << EBdigis->size() ;
   edm::LogInfo("EcalUncalibRecHitInfo") << "EcalWeightUncalibRecHitProducer: total # EEdigis: " << EEdigis->size() ;
//      }


   // collection of reco'ed ampltudes to put in the event

   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

   std::vector<double> pedVec;
   std::vector<HepMatrix> weights;
   std::vector<HepSymMatrix> chi2mat;

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

   // put the collection of recunstructed hits in the event
   evt.put( EBuncalibRechits, EBhitCollection_ );
   evt.put( EEuncalibRechits, EEhitCollection_ );
}

