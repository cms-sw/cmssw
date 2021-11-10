// system include files
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

// user include files
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class SiStripBadFiberBuilder : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadFiberBuilder(const edm::ParameterSet&);
  ~SiStripBadFiberBuilder() override;

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  bool printdebug_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BadComponentList_;
};

SiStripBadFiberBuilder::SiStripBadFiberBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripBadStrip>(iConfig) {
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  BadComponentList_ = iConfig.getUntrackedParameter<Parameters>("BadComponentList");
}

SiStripBadFiberBuilder::~SiStripBadFiberBuilder() = default;

std::unique_ptr<SiStripBadStrip> SiStripBadFiberBuilder::getNewObject() {
  edm::LogInfo("SiStripBadFiberBuilder") << "... creating dummy SiStripBadStrip Data" << std::endl;

  auto obj = std::make_unique<SiStripBadStrip>();

  std::stringstream ss;
  for (Parameters::iterator iBadComponent = BadComponentList_.begin(); iBadComponent != BadComponentList_.end();
       ++iBadComponent) {
    uint32_t BadModule_ = iBadComponent->getParameter<uint32_t>("BadModule");
    std::vector<uint32_t> BadApvList_ = iBadComponent->getParameter<std::vector<uint32_t> >("BadApvList");

    std::vector<unsigned int> theSiStripVector;

    unsigned short firstBadStrip = 0, NconsecutiveBadStrips = 0;
    unsigned int theBadStripRange;

    for (std::vector<uint32_t>::const_iterator is = BadApvList_.begin(); is != BadApvList_.end(); ++is) {
      firstBadStrip = (*is) * 128;
      NconsecutiveBadStrips = 128;

      theBadStripRange = obj->encode(firstBadStrip, NconsecutiveBadStrips);

      if (printdebug_)
        ss << "detid " << BadModule_ << " \t"
           << " firstBadStrip " << firstBadStrip << "\t "
           << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
           << " packed integer " << std::hex << theBadStripRange << std::dec << std::endl;

      theSiStripVector.push_back(theBadStripRange);
    }

    SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(BadModule_, range))
      edm::LogError("SiStripBadFiberBuilder") << "[SiStripBadFiberBuilder::analyze] detid already exists" << std::endl;
  }
  if (printdebug_)
    edm::LogInfo("SiStripBadFiberBuilder") << ss.str();

  return obj;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripBadFiberBuilder);
