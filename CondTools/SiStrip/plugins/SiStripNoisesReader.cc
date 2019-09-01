#include "CondTools/SiStrip/plugins/SiStripNoisesReader.h"

using namespace cms;

SiStripNoisesReader::SiStripNoisesReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

SiStripNoisesReader::~SiStripNoisesReader() {}

void SiStripNoisesReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiStripNoises> SiStripNoises_;
  iSetup.get<SiStripNoisesRcd>().get(SiStripNoises_);
  edm::LogInfo("SiStripNoisesReader") << "[SiStripNoisesReader::analyze] End Reading SiStripNoises" << std::endl;

  std::vector<uint32_t> detid;
  SiStripNoises_->getDetIds(detid);
  edm::LogInfo("Number of detids ") << detid.size() << std::endl;

  if (printdebug_)
    for (size_t id = 0; id < detid.size() && id < printdebug_; id++) {
      SiStripNoises::Range range = SiStripNoises_->getRange(detid[id]);

      int strip = 0;
      for (int it = 0; it < (range.second - range.first) * 8 / 9; it++) {
        edm::LogInfo("SiStripNoisesReader") << "detid " << detid[id] << " \t"
                                            << " strip " << strip++ << " \t" << SiStripNoises_->getNoise(it, range)
                                            << " \t"
                                            //<< SiStripNoises_->getDisable(it,range)   << " \t"
                                            << std::endl;
      }
    }
}
