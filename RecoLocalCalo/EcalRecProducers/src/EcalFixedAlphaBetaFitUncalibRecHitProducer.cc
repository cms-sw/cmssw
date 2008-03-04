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
#include "DataFormats/Common/interface/Handle.h"
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

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

EcalFixedAlphaBetaFitUncalibRecHitProducer::EcalFixedAlphaBetaFitUncalibRecHitProducer(const edm::ParameterSet& ps) {

  EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
  EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
  EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");

  //AmplThrEB_ =  ps.getParameter<double> ("MinAmplBarrel")
  //AmplThrEE_ =  ps.getParameter<double> ("MinAmplEndcap")

  produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
  produces< EEUncalibratedRecHitCollection >(EEhitCollection_);

  
  alpha_= ps.getParameter<double>("alpha");
  beta_= ps.getParameter<double>("beta");

   alphabetaFilename_= ps.getUntrackedParameter<std::string>("AlphaBetaFilename","NOFILE");
   useAlphaBetaArray_=setAlphaBeta();//set crystalwise values of alpha and beta
   if(!useAlphaBetaArray_){edm::LogInfo("EcalUncalibRecHitError") << " No alfa-beta file found. Using the deafult values.";}

   algoEB_.SetMinAmpl( ps.getParameter<double> ("MinAmplBarrel") );
   algoEE_.SetMinAmpl(  ps.getParameter<double> ("MinAmplEndcap") );

   bool dyn_pede = ps.getParameter<bool>("UseDynamicPedestal");
   algoEB_.SetDynamicPedestal(dyn_pede);
   algoEE_.SetDynamicPedestal(dyn_pede);

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
     evt.getByLabel( EBdigiCollection_, pEBDigis);
     //     evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the EB product
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size();
   } catch (...) {
     //     std::cerr << "Error! can't get the product for EB: " << EBdigiCollection_.c_str() << std::endl;
   }
   try {//Endcap
     evt.getByLabel( EEdigiCollection_, pEEDigis);
     //     evt.getByLabel( digiProducer_, pEEDigis);
     EEdigis = pEEDigis.product(); // get a ptr to the EE product
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EEdigis: " << EEdigis->size() ;
   } catch (...) {
     //     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product for EE: " << EEdigiCollection_.c_str() ;
   }

   // Gain Ratios
   LogDebug("EcalUncalibRecHitDebug") << "fetching gainRatios....";
   edm::ESHandle<EcalGainRatios> pRatio;
   es.get<EcalGainRatiosRcd>().get(pRatio);
   const EcalGainRatioMap& gainMap = pRatio.product()->getMap(); // map of gain ratios


   // fetch the pedestals from the cond DB via EventSetup
   LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   const EcalPedestalsMap & pedMap = pedHandle.product()->getMap(); // map of pedestals
   LogDebug("EcalUncalibRecHitDebug") << "done." ;

   // EE and EB collections of reco'ed ampltudes to put in the event
   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

//    std::vector<HepMatrix> weights;
//    std::vector<HepSymMatrix> chi2mat;

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal

   EcalGainRatioMap::const_iterator gainIter; // gain iterator
   EcalMGPAGainRatio aGain; // gain object for a single xtal

   //loop over EB digis
   if( EBdigis ){
     for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {

       // find pedestals for this channel
       LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EBDetId(itdg->id()) ;
       pedIter = pedMap.find(itdg->id());
       if( pedIter != pedMap.end() ) {
	 aped = (*pedIter);
       } else {
	 edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << EBDetId(itdg->id()) 
						 << "\n  no uncalib rechit will be made for this digi!"
	   ;
	 continue;
       }
       double pedVec[3];
       pedVec[0]=aped.mean_x12;pedVec[1]=aped.mean_x6;pedVec[2]=aped.mean_x1;
       
       // find gain ratios
       LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EBDetId(itdg->id()) ;
       gainIter = gainMap.find(itdg->id());
       if( gainIter != gainMap.end() ) {
	 aGain = (*gainIter);
       } else {
	 edm::LogError("EcalUncalibRecHitError") << "error!! could not find gain ratios for channel: " << EBDetId(itdg->id())
						 << "\n  no uncalib rechit will be made for this digi!"
	   ;
	 continue;
       }
       double gainRatios[3];
       gainRatios[0]=1.;gainRatios[1]=aGain.gain12Over6();gainRatios[2]=aGain.gain6Over1()*aGain.gain12Over6();

       double a,b;

       // Define Alpha and Beta either by stored values or by default universal values
       if (useAlphaBetaArray_){
	 if ( alphaBetaValues_[EBDetId(itdg->id()).ism()-1].size()!=0){
	   a=alphaBetaValues_[EBDetId(itdg->id()).ism()-1][EBDetId(itdg->id()).ic()-1].first;
	   b=alphaBetaValues_[EBDetId(itdg->id()).ism()-1][EBDetId(itdg->id()).ic()-1].second;
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

       EcalUncalibratedRecHit aHit =  algoEB_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
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

       // find pedestals for this channel
       LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EEDetId(itdg->id()) ;
       pedIter = pedMap.find(itdg->id());
       if( pedIter != pedMap.end() ) {
	 aped = (*pedIter);
       } else {
	 edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << EEDetId(itdg->id()) 
						 << "\n  no uncalib rechit will be made for this digi!"
	   ;
	 continue;
       }
       double pedVec[3];
       pedVec[0]=aped.mean_x12;pedVec[1]=aped.mean_x6;pedVec[2]=aped.mean_x1;
       
       // find gain ratios
       LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EEDetId(itdg->id()) ;
       gainIter = gainMap.find(itdg->id());
       if( gainIter != gainMap.end() ) {
	 aGain = (*gainIter);
       } else {
	 edm::LogError("EcalUncalibRecHitError") << "error!! could not find gain ratios for channel: " << EEDetId(itdg->id())
						 << "\n  no uncalib rechit will be made for this digi!"
	   ;
	 continue;
       }
       double gainRatios[3];
       gainRatios[0]=1.;gainRatios[1]=aGain.gain12Over6();gainRatios[2]=aGain.gain6Over1()*aGain.gain12Over6();

       EcalUncalibratedRecHit aHit =  algoEE_.makeRecHit(*itdg, pedVec, gainRatios, 0 , 0);
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

