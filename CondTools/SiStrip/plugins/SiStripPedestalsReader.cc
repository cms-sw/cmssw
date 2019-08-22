#include "CondTools/SiStrip/plugins/SiStripPedestalsReader.h"

using namespace std;
using namespace cms;

SiStripPedestalsReader::SiStripPedestalsReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

SiStripPedestalsReader::~SiStripPedestalsReader() {}

void SiStripPedestalsReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiStripPedestals> SiStripPedestals_;
  iSetup.get<SiStripPedestalsRcd>().get(SiStripPedestals_);
  edm::LogInfo("SiStripPedestalsReader") << "[SiStripPedestalsReader::analyze] End Reading SiStripPedestals"
                                         << std::endl;

  std::vector<uint32_t> detid;
  SiStripPedestals_->getDetIds(detid);
  edm::LogInfo("Number of detids ") << detid.size() << std::endl;

  if (printdebug_)
    for (size_t id = 0; id < detid.size() && id < printdebug_; id++) {
      SiStripPedestals::Range range = SiStripPedestals_->getRange(detid[id]);

      int strip = 0;
      for (int it = 0; it < (range.second - range.first) * 8 / 10; it++) {
        edm::LogInfo("SiStripPedestalsReader")
            << "detid  " << detid[id] << " \t"
            << " strip " << strip++ << " \t" << SiStripPedestals_->getPed(it, range) << " \t" << std::endl;
      }
    }
}
