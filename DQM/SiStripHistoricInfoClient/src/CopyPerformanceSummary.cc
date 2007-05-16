#include "DQM/SiStripHistoricInfoClient/interface/CopyPerformanceSummary.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/RunID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiStripPerformanceSummaryRcd.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include <string>
#include <memory>

//---- default constructor / destructor
CopyPerformanceSummary::CopyPerformanceSummary(const edm::ParameterSet& iConfig) {}
CopyPerformanceSummary::~CopyPerformanceSummary() {}

//---- called each event
void CopyPerformanceSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if(firstEventInRun){
    firstEventInRun=false;
  }
  ++nevents;
}

//---- called each BOR
void CopyPerformanceSummary::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  edm::ESHandle<SiStripPerformanceSummary> tkperf;
  iSetup.get<SiStripPerformanceSummaryRcd>().get(tkperf);
  pSummary_ = new SiStripPerformanceSummary(*tkperf.product());
  firstEventInRun=true;
}

//---- called each EOR
void CopyPerformanceSummary::endRun(const edm::Run& run , const edm::EventSetup& iSetup){
  firstEventInRun=false;
  pSummary_->print();
  writeToDB(run);
}

//-----------------------------------------------------------------------------------------------
void CopyPerformanceSummary::beginJob(const edm::EventSetup&) {
  nevents = 0;
}

//-----------------------------------------------------------------------------------------------
void CopyPerformanceSummary::endJob() {
}

//-----------------------------------------------------------------------------------------------
void CopyPerformanceSummary::writeToDB(const edm::Run& run) const {
  unsigned int l_run  = run.id();
  std::cout<<"CopyPerformanceSummary::writeToDB()  run="<<l_run<<std::endl;
  //now write SiStripPerformanceSummary data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    try{
      if( mydbservice->isNewTagRequest("SiStripPerformanceSummaryRcd") ){
        mydbservice->createNewIOV<SiStripPerformanceSummary>(pSummary_,mydbservice->endOfTime(),"SiStripPerformanceSummaryRcd");
      } else {
        mydbservice->appendSinceTime<SiStripPerformanceSummary>(pSummary_,mydbservice->currentTime(),"SiStripPerformanceSummaryRcd");
      }
    }catch(const cond::Exception& er){
      edm::LogError("writeToDB")<<er.what()<<std::endl;
    }catch(const std::exception& er){
      edm::LogError("writeToDB")<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      edm::LogError("writeToDB")<<"Funny error"<<std::endl;
    }
  }else{
    edm::LogError("writeToDB")<<"Service is unavailable"<<std::endl;
  }
}
