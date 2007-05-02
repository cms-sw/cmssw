// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripGainCalculator
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainCalculator.cc,v 1.3 2007/02/21 11:13:07 dkcira Exp $


#include "CalibTracker/SiStripChannelGain/interface/SiStripGainCalculator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"


#include "FWCore/Utilities/interface/Exception.h"
//#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <cstdlib>
#include <string>


using namespace cms;


SiStripGainCalculator::SiStripGainCalculator(const edm::ParameterSet& iConfig) : runNumber_(0) {

  edm::LogInfo("SiStripGainCalculator::SiStripGainCalculator()") << std::endl;
  SiStripApvGain_ = new SiStripApvGain();

}


SiStripGainCalculator::~SiStripGainCalculator(){ 

  edm::LogInfo("SiStripGainCalculator::~SiStripGainCalculator()") << std::endl;

}


void SiStripGainRandomCalculator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

   edm::LogInfo("SiStripGainCalculator::SiStripGainCalculator");

   static bool first = true;
   if (first){
     sinceTime_= iEvent.time() ;
     edm::LogInfo("SiStripGainCalculator::SiStripGainCalculator")<< sinceTime_;

     first=false;
   }


   // here is the call to the concrete analyzers

   algoAnalyze(iEvent, iSetup);


}


void SiStripGainCalculator::beginJob(const edm::EventSetup& iSetup){
 
  edm::LogInfo("SiStripGainCalculator::beginJob") << std::endl;


}


void SiStripGainCalculator::beginRun(const edm::Run & run, const edm::EventSetup & es){

  edm::LogInfo("SiStripGainCalculator::beginRun") << std::endl;

  runNumber_=run.run(); 

  edm::LogInfo("RunNumber: ") << runNumber_<< std::endl;

}




void SiStripGainCalculator::endJob() {


  edm::LogInfo("SiStripGainCalculator") << "... creating dummy SiStripApvGain Data for Run " << runNumber_ << "\n " << std::endl;

  if(! SiStripApvGain_) {
    edm::LogError("SiStripGainCalculator: gain object has not been set...storing no data on DB") ;
    return;
  }
  

  //And now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    try{
      if( mydbservice->isNewTagRequest("SiStripApvGainRcd") ){

	edm::LogInfo("SiStripGainCalculator") << "first request for storing objects with Tag SiStripApvGainRcd. The IOV of this payload is from time 0 to infinity..  " << std::endl;

	mydbservice->createNewIOV<SiStripApvGain>(SiStripApvGain_,mydbservice->endOfTime(),"SiStripApvGainRcd");      
      } else {
	//	mydbservice->appendSinceTime<SiStripApvGain>(SiStripApvGain_,mydbservice->currentTime(),"SiStripApvGainRcd"); 

      edm::LogInfo("SiStripGainCalculator") << "appending a new object to an existing tag. IOV has start time the time of first event analyzed in this job " << std::endl;
	mydbservice->appendSinceTime<SiStripApvGain>(SiStripApvGain_,sinceTime_,"SiStripApvGainRcd");      
      }
    }catch(const cond::Exception& er){
      edm::LogError("SiStripGainCalculator")<<er.what()<<std::endl;
    }catch(const std::exception& er){
      edm::LogError("SiStripGainCalculator")<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      edm::LogError("SiStripGainCalculator")<<"Funny error"<<std::endl;
    }
  }else{
    edm::LogError("SiStripGainCalculator")<<"Service is unavailable"<<std::endl;
  }



}


SiStripApvGain * SiStripGainCalculator::gainCalibrationPointer(){

  return SiStripApvGain_;

}
