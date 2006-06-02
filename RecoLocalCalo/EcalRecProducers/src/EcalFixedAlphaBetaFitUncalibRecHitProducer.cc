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
#include <fstream>

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

 alpha_= ps.getUntrackedParameter<double>("alpha",1.2);
   beta_= ps.getUntrackedParameter<double>("beta",1.7);

   alphabetaFilename_= ps.getUntrackedParameter<std::string>("AlphaBetaFilename","NOFILE");
   useAlphaBetaArray_=setAlphaBeta();//set crystalwise values of alpha and beta
   if(!useAlphaBetaArray_){edm::LogInfo("EcalUncalibRecHitError") << " No alfa-beta file found. Using the deafult values.";}
}

EcalFixedAlphaBetaFitUncalibRecHitProducer::~EcalFixedAlphaBetaFitUncalibRecHitProducer() {
}

//Sets the alphaBetaValues_ vectors by the values provided in alphabetaFilename_
bool EcalFixedAlphaBetaFitUncalibRecHitProducer::setAlphaBeta(){
  std::ifstream file(alphabetaFilename_.c_str());
  if (! file.is_open())
    return false;

  alphaBetaValues_.resize(36);

  char buffer[100];
  int sm, cry,ret;
  float a,b;
  std::pair<double,double> p(-1,-1);

  while( ! file.getline(buffer,100).eof() ){
    ret=sscanf(buffer,"%d %d %f %f", &sm, &cry, &a, &b);
    if ((ret!=4)||
        (sm<=0) ||(sm>36)||
        (cry<=0)||(cry>1700)){
      // send warning
      continue;
    }

    if (alphaBetaValues_[sm-1].size()==0){
      alphaBetaValues_[sm-1].resize(1700,p);
    }
    alphaBetaValues_[sm-1][cry-1].first = a;    
    alphaBetaValues_[sm-1][cry-1].second = b;

  }

  file.close();
  return true;
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

       double a,b;

       // Define Alpha and Beta either by stored values or by default universal values
       if (useAlphaBetaArray_){
	 if ( alphaBetaValues_[itdg->id().ism()-1].size()!=0){
	   a=alphaBetaValues_[itdg->id().ism()-1][itdg->id().ic()-1].first;
	   b=alphaBetaValues_[itdg->id().ism()-1][itdg->id().ic()-1].second;
	   if ((a==-1)&&(b==-1)){
	     a=alpha_;
	     b=beta_;
	   }
	 }else{
	   a=alpha_;
	   b=beta_;
	 }
       }else{
	 a=alpha_;
	 b=beta_;
       }
     
       algoEB_.SetAlphaBeta(a,b);

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
       //FIX ME load in a and b from a file
       algoEE_.SetAlphaBeta(alpha_,beta_);
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

