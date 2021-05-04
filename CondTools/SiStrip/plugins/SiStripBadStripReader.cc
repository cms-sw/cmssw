#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include <iostream>
#include <cstdio>
#include <sys/time.h>

class SiStripBadStripReader : public edm::EDAnalyzer {
public:
  explicit SiStripBadStripReader(const edm::ParameterSet& iConfig)
      : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)), badStripToken_(esConsumes()) {}

  ~SiStripBadStripReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  edm::ESGetToken<SiStripBadStrip, SiStripBadStripRcd> badStripToken_;
};

SiStripBadStripReader::~SiStripBadStripReader() {}

void SiStripBadStripReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& badStrip = iSetup.getData(badStripToken_);
  edm::LogInfo("SiStripBadStripReader") << "[SiStripBadStripReader::analyze] End Reading SiStripBadStrip" << std::endl;

  std::vector<uint32_t> detid;
  badStrip.getDetIds(detid);

  if (printdebug_)
    for (size_t id = 0; id < detid.size(); id++) {
      SiStripBadStrip::Range range = badStrip.getRange(detid[id]);

      for (int it = 0; it < range.second - range.first; it++) {
        unsigned int value = (*(range.first + it));
        edm::LogInfo("SiStripBadStripReader") << "detid " << detid[id] << " \t"
                                              << " firstBadStrip " << badStrip.decode(value).firstStrip << "\t "
                                              << " NconsecutiveBadStrips " << badStrip.decode(value).range << "\t "
                                              << " flag " << badStrip.decode(value).flag << "\t "
                                              << " packed integer " << std::hex << value << std::dec
                                              << "\t "
                                              //<< badStrip.getBadStrips(range)     << " \t"
                                              << std::endl;
      }
    }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBadStripReader);
