/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytical fit method
 *
  *  $Id: EcalAnalFitUncalibRecHitProducer.cc,v 1.4 2005/10/25 14:08:30 rahatlou Exp $
  *  $Date: 2005/10/25 14:08:30 $
  *  $Revision: 1.4 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalAnalFitUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

//#include "CLHEP/Matrix/Matrix.h"
//#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

EcalAnalFitUncalibRecHitProducer::EcalAnalFitUncalibRecHitProducer(const edm::ParameterSet& ps) {

   digiCollection_ = ps.getParameter<std::string>("digiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   hitCollection_  = ps.getParameter<std::string>("hitCollection");
   nMaxPrintout_   = ps.getUntrackedParameter<int>("nMaxPrintout",10);
   produces< EcalUncalibratedRecHitCollection >(hitCollection_);
   nEvt_ = 0; // reset local event counter
}

EcalAnalFitUncalibRecHitProducer::~EcalAnalFitUncalibRecHitProducer() {
}

void
EcalAnalFitUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   nEvt_++;

   Handle< EBDigiCollection > pDigis;
   try {
     //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
     evt.getByLabel( digiProducer_, pDigis);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product " << digiCollection_.c_str() << std::endl;
   }
   const EBDigiCollection* digis = pDigis.product(); // get a ptr to the product
   if(!counterExceeded()) std::cout << "EcalAnalFitUncalibRecHitProducer: total # digis: " << digis->size() << std::endl;


   // collection of reco'ed ampltudes to put in the event
   std::auto_ptr< EcalUncalibratedRecHitCollection > uncalibRechits( new EcalUncalibratedRecHitCollection );

   std::vector<double> pedVec;
   std::vector<HepMatrix> weights;
   std::vector<HepSymMatrix> chi2mat;
   for(EBDigiCollection::const_iterator itdg = digis->begin(); itdg != digis->end(); ++itdg) {

     EcalUncalibratedRecHit aHit =
          algo_.makeRecHit(*itdg, pedVec, weights, chi2mat);
     uncalibRechits->push_back( aHit );

     if(aHit.amplitude()>0. && !counterExceeded() ) {
        std::cout << "EcalAnalFitUncalibRecHitProducer: processed EBDataFrame with id: "
                  << itdg->id() << "\n"
                  << "uncalib rechit amplitude: " << aHit.amplitude()
                  << std::endl;
     }
   }

   // put the collection of recunstructed hits in the event
   evt.put( uncalibRechits, hitCollection_ );
}

