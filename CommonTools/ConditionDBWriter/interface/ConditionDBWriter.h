#ifndef CommonTools_ConditionDBWriter_ConditionDBWriter_h
#define CommonTools_ConditionDBWriter_ConditionDBWriter_h
// -*- C++ -*-
//
// Package:    ConditionDBWriter
// Class:      ConditionDBWriter
// 
// \class ConditionDBWriter 
//
//  Description: 

/**
 *  Implementation:
 *
 *  This class can be very useful whenever a CMSSW application needs to store data
 *  to the offline DB. Typically such applications require access to event data 
 *  and/or need to be notified about the start of Run, Lumi section in order 
 *  to set a correct Interval Of Validity (IOV) for the data they have to store.
 *  Therefore the FWK EDAnalyzer is an excellent candidate for the implementation
 *  of such applications; this is the reason why this class inherits from 
 *  the EDAnalyzer class. 
 *
 *  The user class should inherit from this class. 
 *  The templated type must be the type of the object that
 *  has to be written on the DB (e.g. MyCalibration). Examples of use of
 *  this class can be found in package CalibTracker/SiStripChannelGain. Have a
 *  look also at the test/ directory for examples of full cfg files. 
 *
 *  The user must implement in his derived class the abstract method below
 *
 *  virtual MyCalibration * getNewObject()=0;
 *
 *  in this method, the user must create a new instance of the DB object and 
 *  return a pointer to it. The object must be created with "new" and never 
 *  be deleted by the user: it will be the FWK that takes control over it.
 *
 *  The user can optionally implement the following methods 
 *
 *    //Will be called at the beginning of the job
 *    virtual void algoBeginJob(const edm::EventSetup&){};
 *    //Will be called at the beginning of each run in the job
 *    virtual void algoBeginRun(const edm::Run &, const edm::EventSetup &){};
 *    //Will be called at the beginning of each luminosity block in the run
 *    virtual void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};
 *    //Will be called at every event
 *    virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&){};
 *    //Will be called at the end of each run in the job
 *    virtual void algoEndRun(const edm::Run &, const edm::EventSetup &){};
 *    //Will be called at the end of the job
 *    virtual void algoEndJob(){};
 *
 *  where he can access information needed to build his object. For instance, if
 *  he is computing a calibration that is computed as the mean of a certain
 *  quantity that varies from event to event, he will implement the algoAnalyze 
 *  method.
 *
 *  The important part is the IOV setting. The advantage of using this class is 
 *  that this setting is performed almost automatically: the only thing
 *  that the user has to do is to pass prescriptions about the IOV setting
 *  in the configuration of his module. A typical
 *  configuration is as follows:
 *
 *
 *        module prod =  SiStripGainRandomCalculator {
 *
 *        #parameters of the derived class
 *  		double MinPositiveGain = 0.1
 *  		double MeanGain    = 1
 *  		double SigmaGain   = 0
 *                  untracked bool   printDebug = true
 *
 *        #parameters of the base class
 *  		string IOVMode	     = "Run"
 *  		bool SinceAppendMode = true
 *  		string Record        = "SiStripApvGainRcd"
 *
 *                 }
 *
 *  Two subsets of parameters can be found. The first subset contains the specific
 *  parameters of the user class, which is called in this case 
 *  SiStripGainRandomCalculator. The second subset contains the parameters of
 *  the base class. These are the following:
 *
 *  1) string IOVMode
 *
 *  4 possible values can be given: "Job", "Run", "LumiBlock" and "AlgoDriven"
 *  This card determines the length of the IOV. In other words, together with  
 *  the number of Lumysections/runs the user has decided to run his application,
 *  this card determines the number of objects that will be stored on the DB
 *  (the getNewObject method will be called as many times).
 *  For example if the user is running on the events of one Run, which has 
 *  10 luminosity sections and chooses the "LumiBlock" mode, then 10 objects
 *  with corresponding IOV will be written. If the "Job" mode is chosen, only one 
 *  object will be stored irrespective of the dataset on which the user is 
 *  running.
 *  The  "AlgoDriven" option is special. If this choice is made, then it is 
 *  up to the user to tell in the code when the getNewObject method must be 
 *  called. This can be done by calling the method  below  void storeOnDbNow()
 *  must be invoked whenever a certain condition that justifies the start/end
 *  of an IOV is met.
 *
 *  2) bool SinceAppendMode
 *
 *  obsolete option
 *  now ONLY Since append mode is supported
 *
 *
 * 
 *      WARNING: due to the policy of the framework, objects SHALL be stored
 *      in IOV chronological order. If you have 10 runs, then execute your application starting from run 1 and not for example in two steps: first from Run 6 to Run 10 and then from Run 1 to Run 6.
 *
 *
 *  3)string Record 
 *
 *  this is the eventsetup record of your object.
 *
 * Note that the setDoStore method changes the doStore parameter read from configuration file.
 * This is sometimes needed e.g. to avoid filling bad payloads to the database.
 *
 */

//


//
// Original Author:  Giacomo Bruno
//         Created:  May 23 10:04:31 CET 2007
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

  explicit ConditionDBWriter(const edm::ParameterSet& iConfig) : LumiBlockMode_(false), RunMode_(false), JobMode_(false), AlgoDrivenMode_(false), Time_(0), setSinceTime_(false), firstRun_(true)
  {
    edm::LogInfo("ConditionDBWriter::ConditionDBWriter()") << std::endl;
    SinceAppendMode_=iConfig.getParameter<bool>("SinceAppendMode");
    std::string IOVMode=iConfig.getParameter<std::string>("IOVMode");
    if (IOVMode==std::string("Job")) JobMode_=true;
    else if (IOVMode==std::string("Run")) RunMode_=true;
    else if (IOVMode==std::string("LumiBlock")) LumiBlockMode_=true;
    else if (IOVMode==std::string("AlgoDriven")) AlgoDrivenMode_=true;
    else  edm::LogError("ConditionDBWriter::ConditionDBWriter(): ERROR - unknown IOV interval write mode...will not store anything on the DB") << std::endl;
    Record_=iConfig.getParameter<std::string>("Record");
    doStore_=iConfig.getParameter<bool>("doStoreOnDB");
    timeFromEndRun_=iConfig.getUntrackedParameter<bool>("TimeFromEndRun", false);
    
    if(! SinceAppendMode_ ) 
      edm::LogError("ConditionDBWriter::endJob(): ERROR - only SinceAppendMode support!!!!");
  }
  
  virtual ~ConditionDBWriter()
  {
    edm::LogInfo("ConditionDBWriter::~ConditionDBWriter()") << std::endl;
  }
  
private:
  
  // method to be implemented by derived class. Must return a pointer to the DB object to be stored, which must have been created with "new". The derived class looses control on it (must not "delete" it at any time in its code!) 
  
  virtual T * getNewObject()=0;
  
  
  // Optional methods that may be implemented (technically "overridden") in the derived classes if needed
  
  //Will be called at the beginning of the job
  virtual void algoBeginJob(const edm::EventSetup&){};
  //Will be called at the beginning of each run in the job
  virtual void algoBeginRun(const edm::Run &, const edm::EventSetup &){};
  //Will be called at the beginning of each luminosity block in the run
  virtual void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};
  //Will be called at every event
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&){};
  //Will be called at the end of each run in the job
  virtual void algoEndRun(const edm::Run &, const edm::EventSetup &){};
  //Will be called at the end of the job
  virtual void algoEndJob(){};

  void beginJob() {}

  void beginRun(const edm::Run & run, const edm::EventSetup &  es)
  {
    if( firstRun_ ) {
      edm::LogInfo("ConditionDBWriter::beginJob") << std::endl;
      if( (JobMode_ || AlgoDrivenMode_) && SinceAppendMode_) setSinceTime_=true;
      algoBeginJob(es);
      firstRun_ = false;
    }
    edm::LogInfo("ConditionDBWriter::beginRun") << std::endl;
    if(RunMode_ && SinceAppendMode_) setSinceTime_=true;
    algoBeginRun(run,es);
  }
  
  void beginLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & iSetup)
  {
    edm::LogInfo("ConditionDBWriter::beginLuminosityBlock") << std::endl;
    if(LumiBlockMode_ && SinceAppendMode_) setSinceTime_=true;
    algoBeginLuminosityBlock(lumiBlock, iSetup);
  }

  void analyze(const edm::Event& event, const edm::EventSetup& iSetup)
  {
    if(setSinceTime_ ){
      setTime(); //set new since time for possible next upload to DB  
      setSinceTime_=false;
    }
    algoAnalyze(event, iSetup);
  }
  
  void endLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & es)
  {
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

  void endRun(const edm::Run & run, const edm::EventSetup & es)
  {
    edm::LogInfo("ConditionDBWriter::endRun") << std::endl;
    
    algoEndRun(run, es);
    
    if(RunMode_){
      
      T * objPointer = getNewObject();
      
      if(objPointer ){
	if( timeFromEndRun_ ) Time_ = run.id().run();
	storeOnDb(objPointer);
      }
      else {
	edm::LogError("ConditionDBWriter::endRun(): ERROR - requested to store on DB on a Run based interval, but received null pointer...will not store anything on the DB") << std::endl;
      }
    }
  }
  
  void endJob()
  {
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

  void storeOnDb(T * objPointer)
  {
    edm::LogInfo("ConditionDBWriter::storeOnDb ")  << std::endl;
    
    setSinceTime_=true;
    
    if(! objPointer) {
      edm::LogError("ConditionDBWriter: Pointer to object has not been set...storing no data on DB") ;
      return;
    }
    
    //And now write  data in DB
    if( !doStore_ ) return;
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if (!  mydbservice.isAvailable() ) {
      edm::LogError("ConditionDBWriter")<<"PoolDBOutputService is unavailable"<<std::endl;
      return;
    }
    
    cond::Time_t since = 
      ( mydbservice->isNewTagRequest(Record_) && !timeFromEndRun_ ) ? mydbservice->beginOfTime() : Time_;

    edm::LogInfo("ConditionDBWriter") << "appending a new object to tag " 
				      <<Record_ <<" in since mode " << std::endl;
    mydbservice->writeOne<T>(objPointer, since, Record_);
  }

  void setTime()
  {
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

  void storeOnDbNow()
  {
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

  /// When set to false the payload will not be written to the db
  void setDoStore(const bool doStore) {doStore_ = doStore;}

private:
  
  bool SinceAppendMode_; // till or since append mode 

  bool LumiBlockMode_; //LumiBlock since/till time
  bool RunMode_; //
  bool JobMode_;
  bool AlgoDrivenMode_;
  bool doStore_;

  std::string Record_;
  cond::Time_t Time_; //time until which the DB object is valid. It is taken from the time of the first event analyzed. The end of the validity is infinity. However as soon as a new DB object with a later start time is inserted, the end time of this one becomes the start time of the new one. 

  bool setSinceTime_;

  bool firstRun_;

  bool timeFromEndRun_;
};

#endif
