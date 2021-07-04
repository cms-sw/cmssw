#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <ext/hash_map>

class SiStripBadModuleByHandBuilder : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadModuleByHandBuilder(const edm::ParameterSet&);
  ~SiStripBadModuleByHandBuilder() override = default;

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

private:
  edm::FileInPath fp_;
  bool printdebug_;
  std::vector<uint32_t> BadModuleList_;
};

SiStripBadModuleByHandBuilder::SiStripBadModuleByHandBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripBadStrip>(iConfig) {
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile));
  BadModuleList_ = iConfig.getUntrackedParameter<std::vector<uint32_t> >("BadModuleList");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
}

std::unique_ptr<SiStripBadStrip> SiStripBadModuleByHandBuilder::getNewObject() {
  const auto detInfo = SiStripDetInfoFileReader::read(fp_.fullPath());

  auto obj = std::make_unique<SiStripBadStrip>();

  unsigned int firstBadStrip = 0;
  unsigned short NconsecutiveBadStrips;
  unsigned int theBadStripRange;

  for (std::vector<uint32_t>::const_iterator it = BadModuleList_.begin(); it != BadModuleList_.end(); ++it) {
    std::vector<unsigned int> theSiStripVector;

    NconsecutiveBadStrips = detInfo.getNumberOfApvsAndStripLength(*it).first * 128;
    theBadStripRange = obj->encode(firstBadStrip, NconsecutiveBadStrips);
    if (printdebug_)
      edm::LogInfo("SiStripBadModuleByHandBuilder")
          << " BadModule " << *it << " \t"
          << " firstBadStrip " << firstBadStrip << "\t "
          << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
          << " packed integer " << std::hex << theBadStripRange << std::dec << std::endl;

    theSiStripVector.push_back(theBadStripRange);
    SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(*it, range))
      edm::LogError("SiStripBadModuleByHandBuilder")
          << "[SiStripBadModuleByHandBuilder::analyze] detid already exists" << std::endl;
  }
  return obj;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBadModuleByHandBuilder);
