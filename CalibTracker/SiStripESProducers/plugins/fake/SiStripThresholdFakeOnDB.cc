
#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripThresholdFakeOnDB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"

SiStripThresholdFakeOnDB::SiStripThresholdFakeOnDB(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripThreshold>::ConditionDBWriter<SiStripThreshold>(iConfig){

  
  edm::LogInfo("SiStripThresholdFakeOnDB::SiStripThresholdFakeOnDB");
}


SiStripThresholdFakeOnDB::~SiStripThresholdFakeOnDB(){

   edm::LogInfo("SiStripThresholdFakeOnDB::~SiStripThresholdFakeOnDB");
}



void SiStripThresholdFakeOnDB::algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup){

  edm::ESHandle<SiStripThreshold> thresholdHandle;

  iSetup.get<SiStripThresholdRcd>().get(thresholdHandle);
  
  edm::LogInfo("SiStripThresholdFakeOnDB") << "[SiStripThresholdFakeOnDB::algoAnalyze] End Reading SiStripThreshold" << std::endl;
  
  threshold_ = new SiStripThreshold(*thresholdHandle);
}


SiStripThreshold * SiStripThresholdFakeOnDB::getNewObject() {
  return threshold_;
}



