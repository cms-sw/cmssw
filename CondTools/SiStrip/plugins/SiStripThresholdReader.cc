#include "CondTools/SiStrip/plugins/SiStripThresholdReader.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"

using namespace std;
using namespace cms;

SiStripThresholdReader::SiStripThresholdReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 3)), thresholdToken_(esConsumes()) {}

SiStripThresholdReader::~SiStripThresholdReader() {}

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
