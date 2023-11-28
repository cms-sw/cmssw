#ifndef CalibTracker_SiStripESProducer_DummyCondDBWriter_h
#define CalibTracker_SiStripESProducer_DummyCondDBWriter_h

// user include files
#include <string>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

template <typename TObject, typename TObjectO, typename TRecord>
class DummyCondDBWriter : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit DummyCondDBWriter(const edm::ParameterSet& iConfig);
  ~DummyCondDBWriter() override;
  void analyze(const edm::Event& e, const edm::EventSetup& es) override{};
  void beginRun(const edm::Run& run, const edm::EventSetup& es) override{};
  void endRun(const edm::Run& run, const edm::EventSetup& es) override;

private:
  edm::ParameterSet iConfig_;
  edm::ESWatcher<TRecord> watcher_;
  edm::ESGetToken<TObject, TRecord> token_;
};

template <typename TObject, typename TObjectO, typename TRecord>
DummyCondDBWriter<TObject, TObjectO, TRecord>::DummyCondDBWriter(const edm::ParameterSet& iConfig)
    : iConfig_(iConfig),
      token_(esConsumes<edm::Transition::EndRun>(
          edm::ESInputTag{"", iConfig.getUntrackedParameter<std::string>("label", "")})) {
  usesResource(cond::service::PoolDBOutputService::kSharedResource);
  edm::LogInfo("DummyCondDBWriter") << "DummyCondDBWriter constructor for typename " << typeid(TObject).name()
                                    << " and record " << typeid(TRecord).name();
}

template <typename TObject, typename TObjectO, typename TRecord>
DummyCondDBWriter<TObject, TObjectO, TRecord>::~DummyCondDBWriter() {
  edm::LogInfo("DummyCondDBWriter") << "DummyCondDBWriter::~DummyCondDBWriter()";
}

template <typename TObject, typename TObjectO, typename TRecord>
void DummyCondDBWriter<TObject, TObjectO, TRecord>::endRun(const edm::Run& run, const edm::EventSetup& es) {
  std::string rcdName = iConfig_.getParameter<std::string>("record");

  if (!watcher_.check(es)) {
    edm::LogInfo("DummyCondDBWriter") << "Not needed to store objects with Record " << rcdName << " at run "
                                      << run.run();
    return;
  }

  auto obj = std::make_unique<TObjectO>(es.getData(token_));
  cond::Time_t Time_;

  //And now write  data in DB
  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if (dbservice.isAvailable()) {
    std::string openIovAt = iConfig_.getUntrackedParameter<std::string>("OpenIovAt", "beginOfTime");
    if (openIovAt == "beginOfTime")
      Time_ = dbservice->beginOfTime();
    else if (openIovAt == "currentTime")
      Time_ = dbservice->currentTime();
    else
      Time_ = iConfig_.getUntrackedParameter<uint32_t>("OpenIovAtTime", 1);

    dbservice->writeOneIOV(*obj, Time_, rcdName);
  } else {
    edm::LogError("DummyCondDBWriter") << "Service is unavailable";
  }
}

#endif
