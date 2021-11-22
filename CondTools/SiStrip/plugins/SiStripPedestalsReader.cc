// system include files
//#include <memory>
#include <iostream>
#include <cstdio>
#include <sys/time.h>

// user include files
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class SiStripPedestalsReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripPedestalsReader(const edm::ParameterSet&);
  ~SiStripPedestalsReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  const edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;
};

using namespace std;
using namespace cms;

SiStripPedestalsReader::SiStripPedestalsReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)), pedestalsToken_(esConsumes()) {}

void SiStripPedestalsReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& pedestals = iSetup.getData(pedestalsToken_);
  edm::LogInfo("SiStripPedestalsReader") << "[SiStripPedestalsReader::analyze] End Reading SiStripPedestals"
                                         << std::endl;

  std::vector<uint32_t> detid;
  pedestals.getDetIds(detid);
  edm::LogInfo("Number of detids ") << detid.size() << std::endl;

  if (printdebug_)
    for (size_t id = 0; id < detid.size() && id < printdebug_; id++) {
      SiStripPedestals::Range range = pedestals.getRange(detid[id]);

      int strip = 0;
      for (int it = 0; it < (range.second - range.first) * 8 / 10; it++) {
        edm::LogInfo("SiStripPedestalsReader")
            << "detid  " << detid[id] << " \t"
            << " strip " << strip++ << " \t" << pedestals.getPed(it, range) << " \t" << std::endl;
      }
    }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripPedestalsReader);
