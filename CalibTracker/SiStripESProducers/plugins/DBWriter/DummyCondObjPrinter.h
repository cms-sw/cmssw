#ifndef CalibTracker_SiStripESProducer_DummyCondObjPrinter_h
#define CalibTracker_SiStripESProducer_DummyCondObjPrinter_h

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include <string>

template <typename TObject, typename TRecord>
class DummyCondObjPrinter : public edm::EDAnalyzer {
public:
  explicit DummyCondObjPrinter(const edm::ParameterSet& iConfig);
  ~DummyCondObjPrinter() override;
  void analyze(const edm::Event& e, const edm::EventSetup& es) override;

private:
  edm::ParameterSet iConfig_;
  edm::ESWatcher<TRecord> watcher_;
  edm::ESGetToken<TObject, TRecord> token_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

template <typename TObject, typename TRecord>
DummyCondObjPrinter<TObject, TRecord>::DummyCondObjPrinter(const edm::ParameterSet& iConfig)
    : iConfig_(iConfig), token_(esConsumes()), tTopoToken_(esConsumes()) {
  edm::LogInfo("DummyCondObjPrinter") << "DummyCondObjPrinter constructor for typename " << typeid(TObject).name()
                                      << " and record " << typeid(TRecord).name() << std::endl;
}

template <typename TObject, typename TRecord>
DummyCondObjPrinter<TObject, TRecord>::~DummyCondObjPrinter() {
  edm::LogInfo("DummyCondObjPrinter") << "DummyCondObjPrinter::~DummyCondObjPrinter()" << std::endl;
}

template <typename TObject, typename TRecord>
void DummyCondObjPrinter<TObject, TRecord>::analyze(const edm::Event& e, const edm::EventSetup& es) {
  if (!watcher_.check(es))
    return;

  const auto& esobj = es.getData(token_);
  const auto tTopo = &es.getData(tTopoToken_);
  std::stringstream sSummary, sDebug;
  esobj.printSummary(sSummary, tTopo);
  esobj.printDebug(sDebug, tTopo);

  //  edm::LogInfo("DummyCondObjPrinter") << "\nPrintSummary \n" << sSummary.str()  << std::endl;
  //  edm::LogWarning("DummyCondObjPrinter") << "\nPrintDebug \n" << sDebug.str()  << std::endl;
  edm::LogPrint("DummyCondObjContentPrinter") << "\nPrintSummary \n" << sSummary.str() << std::endl;
  edm::LogVerbatim("DummyCondObjContentPrinter") << "\nPrintDebug \n" << sDebug.str() << std::endl;
}

#endif
