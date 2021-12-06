// user include files
#include "CondFormats/DataRecord/interface/SiStripSummaryRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// system include files
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <sys/time.h>

class SiStripSummaryReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripSummaryReader(const edm::ParameterSet&);
  ~SiStripSummaryReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  const edm::ESGetToken<SiStripSummary, SiStripSummaryRcd> summaryToken_;
};

SiStripSummaryReader::SiStripSummaryReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)), summaryToken_(esConsumes()) {}

void SiStripSummaryReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& summary = iSetup.getData(summaryToken_);
  edm::LogInfo("SiStripSummaryReader") << "[SiStripSummaryReader::analyze] End Reading SiStripSummary" << std::endl;

  std::vector<uint32_t> detid = summary.getDetIds();

  std::vector<std::string> userDB = summary.getUserDBContent();
  std::stringstream ss;
  for (size_t i = 0; i < userDB.size(); ++i)
    ss << userDB[i] << " \n";

  if (printdebug_)
    for (size_t id = 0; id < detid.size(); id++) {
      ss << "\n\tdetid " << detid[id];
      SiStripSummary::Range range = summary.getRange(detid[id]);

      size_t c = 0;
      for (SiStripSummary::ContainerIterator it = range.first; it != range.second; ++it) {
        ss << "\n\t\t " << userDB[c++] << " " << *it;
      }
    }
  edm::LogInfo("SiStripSummaryReader") << ss.str();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripSummaryReader);
