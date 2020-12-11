#include "CondTools/SiStrip/plugins/SiStripPedestalsReader.h"

using namespace std;
using namespace cms;

SiStripPedestalsReader::SiStripPedestalsReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)), pedestalsToken_(esConsumes()) {}

SiStripPedestalsReader::~SiStripPedestalsReader() {}

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
