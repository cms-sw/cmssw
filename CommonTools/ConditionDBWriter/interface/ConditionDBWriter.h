#ifndef CommonTools_ConditionDBWriter_ConditionDBWriter_h
#define CommonTools_ConditionDBWriter_ConditionDBWriter_h
// -*- C++ -*-
//
// Package:    ConditionDBWriter
// Class:      ConditionDBWriter
// 
/**\class ConditionDBWriter ConditionDBWriter.cc CalibTracker/SiStripChannelGain/src/ConditionDBWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giacomo Bruno
//         Created:  May 23 10:04:31 CET 2007
// $Id: ConditionDBWriter.h,v 1.3 2007/05/09 16:10:13 gbruno Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <cstdlib>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Utilities/interface/Exception.h"
//#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


template< class T >
class ConditionDBWriter : public edm::EDAnalyzer {

public:

  explicit ConditionDBWriter(const edm::ParameterSet& iConfig) : LumiBlockMode_(false), RunMode_(false), JobMode_(false), AlgoDrivenMode_(false), Time_(0), setSinceTime_(false) {

    edm::LogInfo("ConditionDBWriter::ConditionDBWriter()") << std::endl;
    SinceAppendMode_=iConfig.getParameter<bool>("SinceAppendMode");
    std::string IOVMode=iConfig.getParameter<std::string>("IOVMode");
    if (IOVMode==std::string("Job")) JobMode_=true;
    else if (IOVMode==std::string("Run")) RunMode_=true;
    else if (IOVMode==std::string("LumiBlock")) LumiBlockMode_=true;
    else if (IOVMode==std::string("AlgoDriven")) AlgoDrivenMode_=true;
    else  edm::LogError("ConditionDBWriter::ConditionDBWriter(): ERROR - unknown IOV interval write mode...will not store anything on the DB") << std::endl;
    Record_=iConfig.getParameter<std::string>("Record");


  }


  ~ConditionDBWriter(){
  
    edm::LogInfo("ConditionDBWriter::~ConditionDBWriter()") << std::endl;

  }


private:


  // method to be implemented by derived class. Must return a pointer to the DB object to be stored, which must have been created with "new". The derived class looses control on it (must not "delete" it at any time in its code!) 

  virtual T * getNewObject()=0;


  // Optional methods that may be implemented in the derived classes if needed

  //Called at the beginning of the job
  virtual void algoBeginJob(const edm::EventSetup&){};
  //Called at the beginning of each run in the job
  virtual void algoBeginRun(const edm::Run &, const edm::EventSetup &){};
  //Called at the beginning of each luminosity block in the run
  virtual void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};
  //called at every event
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&){};
  //Called at the end of each run in the job
  virtual void algoEndRun(const edm::Run &, const edm::EventSetup &){};
  //Called at the end of the job
  virtual void algoEndJob(){};



  void beginJob(const edm::EventSetup& iSetup){

    edm::LogInfo("ConditionDBWriter::beginJob") << std::endl;
    if( (JobMode_ || AlgoDrivenMode_) && SinceAppendMode_) setSinceTime_=true;
    algoBeginJob(iSetup);

  }


  void beginRun(const edm::Run & run, const edm::EventSetup &  es){

    edm::LogInfo("ConditionDBWriter::beginRun") << std::endl;
    if(RunMode_ && SinceAppendMode_) setSinceTime_=true;
    algoBeginRun(run,es);

  }



  void beginLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & iSetup){

    edm::LogInfo("ConditionDBWriter::beginLuminosityBlock") << std::endl;
    if(LumiBlockMode_ && SinceAppendMode_) setSinceTime_=true;
    algoBeginLuminosityBlock(lumiBlock, iSetup);

  }


  void analyze(const edm::Event& event, const edm::EventSetup& iSetup){
    
    if(SinceAppendMode_ && setSinceTime_ ){
      setTime(); //set new since time for possible next upload to DB  
      setSinceTime_=false;
    }

    algoAnalyze(event, iSetup);     

  }



  void endLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & es){

    edm::LogInfo("ConditionDBWriter::endLuminosityBlock") << std::endl;
    algoEndLuminosityBlock(lumiBlock, es);

    if(LumiBlockMode_){

      T * objPointer = getNewObject();
      
      if(objPointer ){
	storeOnDb(objPointer);
      }
      else {
	edm::LogError("ConditionDBWriter::endLuminosityblock(): ERROR - requested to store on DB on a Lumi Block based interval, but received null pointer...will not store anything on the DB") << std::endl;
      }

    }

  }


  virtual void algoEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};


  void endRun(const edm::Run & run, const edm::EventSetup & es){

    edm::LogInfo("ConditionDBWriter::endRun") << std::endl;

    algoEndRun(run, es);

    if(RunMode_){

      T * objPointer = getNewObject();

      if(objPointer ){
	storeOnDb(objPointer);
      }
      else {
	edm::LogError("ConditionDBWriter::endRun(): ERROR - requested to store on DB on a Run based interval, but received null pointer...will not store anything on the DB") << std::endl;
      }

    }

  }



  void endJob(){

    edm::LogInfo("ConditionDBWriter::endJob") << std::endl;

    algoEndJob();

    if(JobMode_){

      T * objPointer = getNewObject();

      if( objPointer ){
	storeOnDb(objPointer);
      }

      else {

	edm::LogError("ConditionDBWriter::endJob(): ERROR - requested to store on DB on a Job based interval, but received null pointer...will not store anything on the DB") << std::endl;
	
      }

    }

  }


  void storeOnDb(T * objPointer){

    edm::LogInfo("ConditionDBWriter::storeOnDb ")  << std::endl;

    if(! SinceAppendMode_ ) setTime();
    else setSinceTime_=true;

    if(! objPointer) {
      edm::LogError("ConditionDBWriter: Pointer to object has not been set...storing no data on DB") ;
      return;
    }
  

    //And now write  data in DB
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
    if( mydbservice.isAvailable() ){

      try{
	
	bool tillDone=false;

	//if first time tag is populated
	if( mydbservice->isNewTagRequest(Record_) ){
	
	  edm::LogInfo("ConditionDBWriter") << "first request for storing objects with Record "<< Record_ << std::endl;
	  
	  if(SinceAppendMode_) {
	    //	  edm::LogInfo("ConditionDBWriter") << "appending a new DUMMY object to new tag "<<Record_<<" in since mode " << std::endl;
	    //	  mydbservice->createNewIOV<T>(new T(), mydbservice->endOfTime(), Record_);
	    edm::LogInfo("ConditionDBWriter") << "appending a new object to existing tag " <<Record_ <<" in since mode " << std::endl;
	    mydbservice->createNewIOV<T>(objPointer, mydbservice->endOfTime(), Record_);

	    // mydbservice->appendSinceTime<T>(objPointer, Time_, Record_); 
	  }
	  else{
	    edm::LogInfo("ConditionDBWriter") << "appending a new object to new tag "<<Record_<< " in till mode " << std::endl;
	    mydbservice->createNewIOV<T>(objPointer, Time_, Record_);      
	    tillDone=true;
	  }
	  
	} 
	else {
	  
	  if(SinceAppendMode_){
	    edm::LogInfo("ConditionDBWriter") << "appending a new object to existing tag " <<Record_ <<" in since mode " << std::endl;
	    mydbservice->appendSinceTime<T>(objPointer, Time_, Record_); 
	  }
	  else if(!tillDone){
	    edm::LogInfo("ConditionDBWriter") << "appending a new object to existing tag "<<Record_ <<" in till mode." << std::endl;
	    //	  mydbservice->appendTillTime<T>(objPointer,Time_,"TRcd");      
	    mydbservice->appendTillTime<T>(objPointer, Time_, Record_);      
	  }

	}

      }catch(const cond::Exception& er){
	edm::LogError("ConditionDBWriter")<<er.what()<<std::endl;
      }catch(const std::exception& er){
	edm::LogError("ConditionDBWriter")<<"caught std::exception "<<er.what()<<std::endl;
      }catch(...){
	edm::LogError("ConditionDBWriter")<<"Funny error"<<std::endl;
      }
    }else{
      edm::LogError("ConditionDBWriter")<<"Service is unavailable"<<std::endl;
    }
    

    
  }



  void setTime(){

    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    
    if( mydbservice.isAvailable() ){
      Time_ = mydbservice->currentTime();
      edm::LogInfo("ConditionDBWriter::setTime: time set to ") << Time_ << std::endl;
    }
    else{
      edm::LogError("ConditionDBWriter::setTime(): PoolDBOutputService is not available...cannot set current time") << std::endl;
    }
    
  }


protected:

  // This method should be called by the derived class only if it support the algodriven mode; this method will trigger a call of  the getNewObject method, but only if algoDrivenMode is chosen

  void storeOnDbNow(){

    T * objPointer = 0;

    if(AlgoDrivenMode_){

      setSinceTime_=true;

      objPointer = getNewObject();
  
      if (!objPointer ) {
	edm::LogError("ConditionDBWriter::storeOnDbNow: ERROR - requested to store on DB a new object (module configuration is algo driven based IOV), but received NULL pointer...will not store anything on the DB") << std::endl;
	return;
      }
      else {storeOnDb(objPointer);}
      
    }
    else {
      
      edm::LogError("ConditionDBWriter::storeOnDbNow(): ERROR - received a direct request from concrete algorithm to store on DB a new object, but module configuration is not to store on DB on an algo driven based interval...will not store anything on the DB") << std::endl;
      return;
    }

  }


  // utility method: it returns the lastly set IOV time (till or since according to what was chosen in the configuration)

  cond::Time_t timeOfLastIOV(){return Time_;}



private:
  
  bool SinceAppendMode_; // till or since append mode 

  bool LumiBlockMode_; //LumiBlock since/till time
  bool RunMode_; //
  bool JobMode_;
  bool AlgoDrivenMode_;

  std::string Record_;
  cond::Time_t Time_; //time until which the DB object is valid. It is taken from the time of the first event analyzed. The end of the validity is infinity. However as soon as a new DB object with a later start time is inserted, the end time of this one becomes the start time of the new one. 

  bool setSinceTime_;


};

#endif
