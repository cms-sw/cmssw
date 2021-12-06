// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"

// system include files
#include <memory>
#include <iostream>
#include <cstdio>
#include <sys/time.h>

class SiStripThresholdReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripThresholdReader(const edm::ParameterSet&);
  ~SiStripThresholdReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  const edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> thresholdToken_;
};

using namespace std;
using namespace cms;

SiStripThresholdReader::SiStripThresholdReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 3)), thresholdToken_(esConsumes()) {}

void SiStripThresholdReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& thresholds = iSetup.getData(thresholdToken_);
  edm::LogInfo("SiStripThresholdReader") << "[SiStripThresholdReader::analyze] End Reading SiStripThreshold"
                                         << std::endl;

  std::vector<uint32_t> detid;
  thresholds.getDetIds(detid);
  edm::LogInfo("Number of detids ") << detid.size() << std::endl;
  if (printdebug_)
    for (size_t id = 0; id < detid.size() && id < printdebug_; id++) {
      SiStripThreshold::Range range = thresholds.getRange(detid[id]);

      //int strip=0;
      float old_clusTh = -1, old_lowTh = -1, old_highTh = -1, old_FirstStrip = -1;
      for (int it = 0; it < 768; it++) {
        SiStripThreshold::Data data = thresholds.getData(it, range);
        std::stringstream ss;
        data.print(ss);
        if (old_clusTh != data.getClusth() || old_lowTh != data.getLth() || old_highTh != data.getHth() ||
            old_FirstStrip != data.getFirstStrip()) {
          edm::LogInfo("SiStripThresholdReader")
              << "detid: " << detid[id] << " \t"
              << "strip: " << it << " \t" << ss.str() << "FirstStrip_and_Hth: " << data.FirstStrip_and_Hth << " \n"
              << std::endl;
          old_lowTh = data.getLth();
          old_highTh = data.getHth();
          old_clusTh = data.getClusth();
          old_FirstStrip = data.getFirstStrip();
        }
      }
    }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripThresholdReader);
