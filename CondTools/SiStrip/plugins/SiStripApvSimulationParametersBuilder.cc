#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvSimulationParameters.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"

class SiStripApvSimulationParametersBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripApvSimulationParametersBuilder(const edm::ParameterSet& iConfig) : m_parametersToken(esConsumes()) {}
  ~SiStripApvSimulationParametersBuilder() override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::ESGetToken<SiStripApvSimulationParameters, SiStripApvSimulationParametersRcd> m_parametersToken;
};

void SiStripApvSimulationParametersBuilder::analyze(const edm::Event&, const edm::EventSetup& evtSetup) {
  // copy; DB service needs non-const pointer but does not take ownership
  auto obj = std::make_unique<SiStripApvSimulationParameters>(evtSetup.getData(m_parametersToken));

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripApvSimulationParametersRcd")) {
      mydbservice->createNewIOV<SiStripApvSimulationParameters>(
          obj.get(), mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripApvSimulationParametersRcd");
    } else {
      mydbservice->appendSinceTime<SiStripApvSimulationParameters>(
          obj.get(), mydbservice->currentTime(), "SiStripApvSimulationParametersRcd");
    }
  } else {
    edm::LogError("SiStripApvSimulationParametersBuilder") << "Service is unavailable";
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripApvSimulationParametersBuilder);
