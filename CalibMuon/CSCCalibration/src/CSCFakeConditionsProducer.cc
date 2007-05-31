// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeMap.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeConditionsProducer.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
//#include "CondFormats/DataRecord/interface/CSCIdentifierRcd.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
//#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"


CSCFakeConditionsProducer::CSCFakeConditionsProducer(const edm::ParameterSet& iConfig)
{
  map_.prefillMap();

setWhatProduced(this,&CSCFakeConditionsProducer::produce);
findingRecord<CSCGainsRcd>();
findingRecord<CSCcrosstalkRcd>();
findingRecord<CSCNoiseMatrixRcd>();
findingRecord<CSCPedestalsRcd>();
}

CSCFakeConditionsProducer::~CSCFakeConditionsProducer::()
{

}

CSCFakeConditionsProducer::ReturnType
CSCFakeConditionsProducer::produce(const GainsRcd& iRecord)
{
   map_.prefillMap();
  
  CSCGains* mydata=new CSCGains(map_.get());
  CSCcrosstalk* mydata=CSCcrosstalk(map_.get());
  CSCNoiseMatrix* mydata=CSCNoiseMatrix(map_.get());
  return mydata;
  
}

void CSCFakeConditionsProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
					       edm::ValidityInterval & oValidity)
{
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
}
