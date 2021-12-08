#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <sstream>

/**
   @class test_FedCablingBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_FedCablingBuilder : public edm::one::EDAnalyzer<> {
public:
  test_FedCablingBuilder(const edm::ParameterSet&) : cablingToken_(esConsumes()) {}
  ~test_FedCablingBuilder() override = default;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> cablingToken_;
};

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
void test_FedCablingBuilder::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlCabling_) << "[test_FedCablingBuilder::" << __func__ << "]"
                       << " Dumping all FED connections...";
  const SiStripFedCabling* fed_cabling = &setup.getData(cablingToken_);
  // fec cabling
  std::stringstream ss;
  ss << "[test_AnalyzeCabling::" << __func__ << "]"
     << " Dumping all SiStripModule objects in FED cabling..." << std::endl
     << fed_cabling;
  LogTrace(mlCabling_) << ss.str();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(test_FedCablingBuilder);
