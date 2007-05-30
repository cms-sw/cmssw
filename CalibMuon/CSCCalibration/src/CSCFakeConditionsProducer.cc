// Framework
//#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include "CondFormats/DataRecord/interface/CSCIdentifierRcd.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"

class CSCFakeConditionsProducer : public edm::ESProducer {
public:
  CSCFakeConditionsProducer(const edm::ParameterSet&);
  ~CSCFakeConditionsProducer() {}

  std::auto_ptr<CSCobject> 
  produceCSCGains(const CSCGainsRcd&) { return std::auto_ptr<CSCobject>(); }
  std::auto_ptr<CSCobject> 
  produceCSCcrosstalk(const CSCcrosstalkRcd&) { return std::auto_ptr<CSCobject>(); }
  std::auto_ptr<CSCobject>
  produceCSCNoiseMatrix(const CSCNoiseMatrixRcd&)  { return std::auto_ptr<CSCobject>(); }
  //  std::auto_ptr<CSCobject>
  //  produceCSCPedestals(const CSCPedestalsRcd&)  { return std::auto_ptr<CSCobject>(); }
};


CSCFakeConditionsProducer::CSCFakeConditionsProducer(const edm::ParameterSet& iConfig)
{

  edm::LogInfo("CSCObjects") << "This is a fake CSC conditions producer";

  setWhatProduced( this, &CSCFakeConditionsProducer::produceCSCGains );
  setWhatProduced( this, &CSCFakeConditionsProducer::produceCSCcrosstalk );
  setWhatProduced( this, &CSCFakeConditionsProducer::produceCSCNoiseMatrix );
  // setWhatProduced( this, &CSCFakeConditionsProducer::produceCSCPedestals);

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CSCFakeConditionsProducer);
