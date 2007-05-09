// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripGainCalculator
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainCalculator.cc,v 1.2 2007/05/04 20:22:35 gbruno Exp $


#include "CalibTracker/SiStripChannelGain/interface/SiStripGainCalculator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
using namespace std;


SiStripGainCalculator::SiStripGainCalculator(const edm::ParameterSet& iConfig) :  LumiBlockMode_(false), RunMode_(false), JobMode_(false), AlgoDrivenMode_(false), Time_(0), setSinceTime_(false) {

  edm::LogInfo("SiStripGainCalculator::SiStripGainCalculator()") << std::endl;
  SinceAppendMode_=iConfig.getParameter<bool>("SinceAppendMode");
  string IOVMode=iConfig.getParameter<string>("IOVMode");
  if (IOVMode==string("Job")) JobMode_=true;
  else if (IOVMode==string("Run")) RunMode_=true;
  else if (IOVMode==string("LumiBlock")) LumiBlockMode_=true;
  else if (IOVMode==string("AlgoDriven")) AlgoDrivenMode_=true;
  else  edm::LogError("SiStripGainCalculator::SiStripGainCalculator(): ERROR - unknown IOV interval write mode...will not store anything on the DB") << std::endl;
  Record_=iConfig.getParameter<string>("Record");


}


SiStripGainCalculator::~SiStripGainCalculator(){ 

  edm::LogInfo("SiStripGainCalculator::~SiStripGainCalculator()") << std::endl;

}




void SiStripGainCalculator::beginJob(const edm::EventSetup& iSetup){
 
  edm::LogInfo("SiStripGainCalculator::beginJob") << std::endl;
  if( (JobMode_ || AlgoDrivenMode_) && SinceAppendMode_) setSinceTime_=true;
  algoBeginJob(iSetup);

}


void SiStripGainCalculator::beginRun(const edm::Run & run, const edm::EventSetup & es){

  edm::LogInfo("SiStripGainCalculator::beginRun") << std::endl;
  if(RunMode_ && SinceAppendMode_) setSinceTime_=true;
  algoBeginRun(run,es);

}




void SiStripGainCalculator::beginLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup& iSetup){
 
  edm::LogInfo("SiStripGainCalculator::beginLuminosityBlock") << std::endl;
  if(LumiBlockMode_ && SinceAppendMode_) setSinceTime_=true;
  algoBeginLuminosityBlock(lumiBlock, iSetup);

}


void SiStripGainCalculator::analyze(const edm::Event & event, const edm::EventSetup& iSetup){


  if(SinceAppendMode_ && setSinceTime_ ){
    setTime(); //set new since time for possible next upload to DB  
    setSinceTime_=false;
  }

  algoAnalyze(event, iSetup);  

}

void SiStripGainCalculator::storeOnDbNow(){

  SiStripApvGain * objPointer = 0;

  if(AlgoDrivenMode_){

    setSinceTime_=true;

    objPointer = getNewObject();
  
    if (!objPointer ) {
      edm::LogError("SiStripGainCalculator::storeOnDbNow: ERROR - requested to store on DB a new object (module configuration is algo driven based IOV), but received NULL pointer...will not store anything on the DB") << std::endl;
      return;
    }
    else {storeOnDb(objPointer);}

  }
  else {
      
    edm::LogError("SiStripGainCalculator::storeOnDbNow(): ERROR - received a direct request from concrete algorithm to store on DB a new object, but module configuration is not to store on DB on an algo driven based interval...will not store anything on the DB") << std::endl;
    return;
  }

}




void SiStripGainCalculator::endLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & es){

  algoEndLuminosityBlock(lumiBlock, es);

  if(LumiBlockMode_){

    SiStripApvGain * objPointer = getNewObject();

    if(objPointer ){
      storeOnDb(objPointer);
    }
    else {
      edm::LogError("SiStripGainCalculator::endLuminosityblock(): ERROR - requested to store on DB on a Lumi Block based interval, but received null pointer...will not store anything on the DB") << std::endl;
    }

  }

}


void SiStripGainCalculator::endRun(const edm::Run & run, const edm::EventSetup & es){

  edm::LogInfo("SiStripGainCalculator::endRun") << std::endl;

  algoEndRun(run, es);

  if(RunMode_){

    SiStripApvGain * objPointer = getNewObject();

    if(objPointer ){
      storeOnDb(objPointer);
    }
    else {
      edm::LogError("SiStripGainCalculator::endRun(): ERROR - requested to store on DB on a Run based interval, but received null pointer...will not store anything on the DB") << std::endl;
    }

  }

}




void SiStripGainCalculator::endJob() {


  edm::LogInfo("SiStripGainCalculator::endJob") << std::endl;

  algoEndJob();

  if(JobMode_){

    SiStripApvGain * objPointer = getNewObject();

    if( objPointer ){
      storeOnDb(objPointer);
    }

    else {

      edm::LogError("SiStripGainCalculator::endJob(): ERROR - requested to store on DB on a Job based interval, but received null pointer...will not store anything on the DB") << std::endl;

    }

  }

}


void SiStripGainCalculator::setTime() {

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    Time_ = mydbservice->currentTime();
    edm::LogInfo("SiStripGainCalculator::setTime: time set to ") << Time_ << std::endl;
  }
  else{
    edm::LogError("SiStripGainCalculator::setTime(): PoolDBOutputService is not available...cannot set current time") << std::endl;
  }

}

void SiStripGainCalculator::storeOnDb(SiStripApvGain * gain) {

  edm::LogInfo("SiStripGainCalculator::storeOnDb ")  << std::endl;

  if(! SinceAppendMode_ ) setTime();
  else setSinceTime_=true;

  if(! gain) {
    edm::LogError("SiStripGainCalculator: gain object has not been set...storing no data on DB") ;
    return;
  }
  

  //And now write  data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){

    try{

      bool tillDone=false;

      //if first time tag is populated
      if( mydbservice->isNewTagRequest(Record_) ){
	
	edm::LogInfo("SiStripGainCalculator") << "first request for storing objects with Record "<< Record_ << std::endl;
	
	if(SinceAppendMode_) {
	  //	  edm::LogInfo("SiStripGainCalculator") << "appending a new DUMMY object to new tag "<<Record_<<" in since mode " << std::endl;
	  //	  mydbservice->createNewIOV<SiStripApvGain>(new SiStripApvGain(), mydbservice->endOfTime(), Record_);
	  edm::LogInfo("SiStripGainCalculator") << "appending a new object to existing tag " <<Record_ <<" in since mode " << std::endl;
	  mydbservice->createNewIOV<SiStripApvGain>(gain, mydbservice->endOfTime(), Record_);

	  // mydbservice->appendSinceTime<SiStripApvGain>(gain, Time_, Record_); 
	}
	else{
	  edm::LogInfo("SiStripGainCalculator") << "appending a new object to new tag "<<Record_<< " in till mode " << std::endl;
	  mydbservice->createNewIOV<SiStripApvGain>(gain, Time_, Record_);      
	  tillDone=true;
	}
	
      } 
      else {

	if(SinceAppendMode_){
	  edm::LogInfo("SiStripGainCalculator") << "appending a new object to existing tag " <<Record_ <<" in since mode " << std::endl;
	  mydbservice->appendSinceTime<SiStripApvGain>(gain, Time_, Record_); 
	}
	else if(!tillDone){
	  edm::LogInfo("SiStripGainCalculator") << "appending a new object to existing tag "<<Record_ <<" in till mode." << std::endl;
	  //	  mydbservice->appendTillTime<SiStripApvGain>(gain,Time_,"SiStripApvGainRcd");      
	  mydbservice->appendTillTime<SiStripApvGain>(gain, Time_, Record_);      
	}

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
