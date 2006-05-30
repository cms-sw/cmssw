/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytic specific fit method
 *   with alfa and beta fixed.
 *
 *  \author A. Ghezzi, Mar 2006
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalFixedAlphaBetaFitUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "CLHEP/Matrix/Matrix.h"
//#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

EcalFixedAlphaBetaFitUncalibRecHitProducer::EcalFixedAlphaBetaFitUncalibRecHitProducer(const edm::ParameterSet& ps) {

  EBdigiCollection_ = ps.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<std::string>("EEdigiCollection");
  digiProducer_   = ps.getParameter<std::string>("digiProducer");
  EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
  EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");

  produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
  produces< EEUncalibratedRecHitCollection >(EEhitCollection_);

}

EcalFixedAlphaBetaFitUncalibRecHitProducer::~EcalFixedAlphaBetaFitUncalibRecHitProducer() {
}

void
EcalFixedAlphaBetaFitUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
   using namespace edm;

   Handle< EBDigiCollection > pEBDigis;
   Handle< EEDigiCollection > pEEDigis;

   const EBDigiCollection* EBdigis =0;
   const EEDigiCollection* EEdigis =0;

   try {//Barrel
     evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the EB product
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size();
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product for EB: " << EBdigiCollection_.c_str() << std::endl;
   }
   try {//Endcap
     evt.getByLabel( digiProducer_, pEEDigis);
     EEdigis = pEEDigis.product(); // get a ptr to the EE product
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EEdigis: " << EEdigis->size() ;
   } catch ( std::exception& ex ) {
     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product for EE: " << EEdigiCollection_.c_str() ;
   }


   // EE and EB collections of reco'ed ampltudes to put in the event
   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

   std::vector<double> pedVec;
   std::vector<HepMatrix> weights;
   std::vector<HepSymMatrix> chi2mat;

   //loop over EB digis
   if( EBdigis ){
     for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {
       algoEB_.SetAlphaBeta(1.2,1.7);
       EcalUncalibratedRecHit aHit =  algoEB_.makeRecHit(*itdg, pedVec, weights, chi2mat);
       EBuncalibRechits->push_back( aHit );
       
     /*
     if(aHit.amplitude()>0. && !counterExceeded() ) {
        std::cout << "EcalFixedAlphaBetaFitUncalibRecHitProducer: processed EBDataFrame with id: "
                  << itdg->id() << "\n"
                  << "uncalib rechit amplitude: " << aHit.amplitude()
                  << std::endl;
     }
     */
     }
   }
   evt.put( EBuncalibRechits, EBhitCollection_ );
   //loop over EE digis
   if( EEdigis ){
     for(EEDigiCollection::const_iterator itdg = EEdigis->begin(); itdg != EEdigis->end(); ++itdg) {
       algoEE_.SetAlphaBeta(1.2,1.7);
       EcalUncalibratedRecHit aHit =  algoEE_.makeRecHit(*itdg, pedVec, weights, chi2mat);
       EEuncalibRechits->push_back( aHit );
       
     /*
     if(aHit.amplitude()>0. && !counterExceeded() ) {
        std::cout << "EcalFixedAlphaBetaFitUncalibRecHitProducer: processed EBDataFrame with id: "
                  << itdg->id() << "\n"
                  << "uncalib rechit amplitude: " << aHit.amplitude()
                  << std::endl;
     }
     */
     }
   }

   // put the collection of reconstructed hits in the event
   evt.put( EEuncalibRechits, EEhitCollection_ );
}

