#ifndef CalibTracker_SiStripESProducer_DummyCondDBWriter_h
#define CalibTracker_SiStripESProducer_DummyCondDBWriter_h

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBCommon/interface/Time.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <string>



template< typename TObject , typename TRecord, typename RecordName>
class DummyCondDBWriter : public edm::EDAnalyzer {

public:

  explicit DummyCondDBWriter(const edm::ParameterSet& iConfig);
  ~DummyCondDBWriter();
  void analyze(const edm::Event& e, const edm::EventSetup&es){};

  void endRun(const edm::Run & run, const edm::EventSetup & es);

 private:
  edm::ParameterSet iConfig_;

};

template< typename TObject , typename TRecord, typename RecordName>
DummyCondDBWriter<TObject,TRecord,RecordName>::DummyCondDBWriter(const edm::ParameterSet& iConfig):iConfig_(iConfig){
  edm::LogInfo("DummyCondDBWriter") << "DummyCondDBWriter constructor for typename " << typeid(TObject).name() << " and record " << typeid(TRecord).name() << std::endl;
}


template< typename TObject , typename TRecord , typename RecordName>
DummyCondDBWriter<TObject,TRecord,RecordName>::~DummyCondDBWriter(){
 edm::LogInfo("DummyCondDBWriter") << "DummyCondDBWriter::~DummyCondDBWriter()" << std::endl;
}

template< typename TObject , typename TRecord , typename RecordName>
void DummyCondDBWriter<TObject,TRecord,RecordName>::endRun(const edm::Run & run, const edm::EventSetup & es){
  edm::ESHandle<TObject> esobj;
  es.get<TRecord>().get( esobj ); 
  TObject *obj= new TObject(*(esobj.product()));
  cond::Time_t Time_;  
  
  //And now write  data in DB
  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if( dbservice.isAvailable() ){

    std::string openIovAt=iConfig_.getUntrackedParameter<std::string>("OpenIovAt","beginOfTime");
    if(openIovAt=="beginOfTime")
      Time_=dbservice->beginOfTime();
    else if (openIovAt=="currentTime")
      dbservice->currentTime();
    else
      Time_=iConfig_.getUntrackedParameter<uint32_t>("OpenIovAtTime",1);
    
    //if first time tag is populated
    if( dbservice->isNewTagRequest(RecordName::name())){
      edm::LogInfo("DummyCondDBWriter") << "first request for storing objects with Record "<< RecordName::name() << " at time " << Time_ << std::endl;
      dbservice->createNewIOV<TObject>(obj, Time_ ,dbservice->endOfTime(), RecordName::name());      
    } else {
      edm::LogInfo("DummyCondDBWriter") << "appending a new object to existing tag " <<RecordName::name() <<" in since mode " << std::endl;
      dbservice->appendSinceTime<TObject>(obj, Time_, RecordName::name()); 
    }    
  } else{
    edm::LogError("SiStripFedCablingBuilder")<<"Service is unavailable"<<std::endl;
  }
}

#endif
