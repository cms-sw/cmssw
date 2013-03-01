#ifndef CalibTracker_SiStripESProducer_DummyCondObjPrinter_h
#define CalibTracker_SiStripESProducer_DummyCondObjPrinter_h

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <string>



template< typename TObject, typename TRecord>
class DummyCondObjPrinter : public edm::EDAnalyzer {

public:

  explicit DummyCondObjPrinter(const edm::ParameterSet& iConfig);
  ~DummyCondObjPrinter();
  void analyze(const edm::Event& e, const edm::EventSetup&es);


 private:
  edm::ParameterSet iConfig_;
  unsigned long long cacheID;
};

template< typename TObject, typename TRecord>
DummyCondObjPrinter<TObject,TRecord>::DummyCondObjPrinter(const edm::ParameterSet& iConfig):iConfig_(iConfig),cacheID(0){
  edm::LogInfo("DummyCondObjPrinter") << "DummyCondObjPrinter constructor for typename " << typeid(TObject).name() << " and record " << typeid(TRecord).name() << std::endl;
}


template< typename TObject, typename TRecord>
DummyCondObjPrinter<TObject,TRecord>::~DummyCondObjPrinter(){
 edm::LogInfo("DummyCondObjPrinter") << "DummyCondObjPrinter::~DummyCondObjPrinter()" << std::endl;
}

template< typename TObject,typename TRecord>
void DummyCondObjPrinter<TObject,TRecord>::analyze(const edm::Event& e, const edm::EventSetup&es){

  if( cacheID == es.get<TRecord>().cacheIdentifier())
    return;
  
  cacheID = es.get<TRecord>().cacheIdentifier();

  edm::ESHandle<TObject> esobj;
  es.get<TRecord>().get( esobj );
  std::stringstream sSummary, sDebug;
  esobj->printSummary(sSummary);
  esobj->printDebug(sDebug);

  //  edm::LogInfo("DummyCondObjPrinter") << "\nPrintSummary \n" << sSummary.str()  << std::endl;
  //  edm::LogWarning("DummyCondObjPrinter") << "\nPrintDebug \n" << sDebug.str()  << std::endl;
  edm::LogPrint("DummyCondObjContentPrinter") << "\nPrintSummary \n" << sSummary.str()  << std::endl;
  edm::LogVerbatim("DummyCondObjContentPrinter") << "\nPrintDebug \n" << sDebug.str()  << std::endl;
}

#endif
