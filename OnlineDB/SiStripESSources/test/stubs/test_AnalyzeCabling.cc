#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>

/**
   @class test_AnalyzeCabling 
   @brief Analyzes FEC (and FED) cabling object(s)
*/
class test_AnalyzeCabling : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  test_AnalyzeCabling(const edm::ParameterSet&) : cablingToken_(esConsumes<edm::Transition::BeginRun>()) {}
  ~test_AnalyzeCabling() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> cablingToken_;
};

// -----------------------------------------------------------------------------
void test_AnalyzeCabling::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

// -----------------------------------------------------------------------------
void test_AnalyzeCabling::endRun(const edm::Run& run, const edm::EventSetup& setup) {}

// -----------------------------------------------------------------------------
void test_AnalyzeCabling::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  using namespace sistrip;

  // fed cabling
  LogTrace(mlCabling_) << "[test_AnalyzeCabling::" << __func__ << "]"
                       << " Dumping all connection objects in FED cabling..." << std::endl;

  const SiStripFedCabling* fed_cabling = &setup.getData(cablingToken_);
  // fec cabling
  SiStripFecCabling fec_cabling(*fed_cabling);
  std::stringstream ss;
  ss << "[test_AnalyzeCabling::" << __func__ << "]"
     << " Dumping all SiStripModule objects in FEC cabling..." << std::endl
     << fec_cabling;
  LogTrace(mlCabling_) << ss.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(test_AnalyzeCabling);
