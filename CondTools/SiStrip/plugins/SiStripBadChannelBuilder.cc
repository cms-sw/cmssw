// system include files
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripBadChannelBuilder : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadChannelBuilder(const edm::ParameterSet&);
  ~SiStripBadChannelBuilder() override;

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  edm::FileInPath fp_;
  bool printdebug_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BadComponentList_;
};

SiStripBadChannelBuilder::SiStripBadChannelBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripBadStrip>(iConfig) {
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile));
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  BadComponentList_ = iConfig.getUntrackedParameter<Parameters>("BadComponentList");
}

SiStripBadChannelBuilder::~SiStripBadChannelBuilder() = default;

std::unique_ptr<SiStripBadStrip> SiStripBadChannelBuilder::getNewObject() {
  edm::LogInfo("SiStripBadChannelBuilder") << "... creating dummy SiStripBadStrip Data" << std::endl;

  auto obj = std::make_unique<SiStripBadStrip>();

  const auto detInfo = SiStripDetInfoFileReader::read(fp_.fullPath());

  for (Parameters::iterator iBadComponent = BadComponentList_.begin(); iBadComponent != BadComponentList_.end();
       ++iBadComponent) {
    uint32_t BadModule_ = iBadComponent->getParameter<uint32_t>("BadModule");
    std::vector<uint32_t> BadChannelList_ = iBadComponent->getParameter<std::vector<uint32_t> >("BadChannelList");

    std::vector<unsigned int> theSiStripVector;
    unsigned int NStrips = detInfo.getNumberOfApvsAndStripLength(BadModule_).first * 128;

    uint32_t lastBad = 999;
    unsigned short firstBadStrip = 0, NconsecutiveBadStrips = 0;
    unsigned int theBadStripRange;

    for (std::vector<uint32_t>::const_iterator is = BadChannelList_.begin(); is != BadChannelList_.end(); ++is) {
      if (*is > NStrips - 1)
        break;
      if (*is != lastBad + 1) {
        //new set

        if (lastBad != 999) {
          //save previous set
          theBadStripRange = obj->encode(firstBadStrip, NconsecutiveBadStrips);

          if (printdebug_)
            edm::LogInfo("SiStripBadChannelBuilder")
                << "detid " << BadModule_ << " \t"
                << " firstBadStrip " << firstBadStrip << "\t "
                << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
                << " packed integer " << std::hex << theBadStripRange << std::dec << std::endl;

          theSiStripVector.push_back(theBadStripRange);
        }

        firstBadStrip = *is;
        NconsecutiveBadStrips = 0;
      }
      NconsecutiveBadStrips++;
      lastBad = *is;
    }

    theBadStripRange = obj->encode(firstBadStrip, NconsecutiveBadStrips);
    if (printdebug_)
      edm::LogInfo("SiStripBadChannelBuilder")
          << "detid " << BadModule_ << " \t"
          << " firstBadStrip " << firstBadStrip << "\t "
          << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
          << " packed integer " << std::hex << theBadStripRange << std::dec << std::endl;

    theSiStripVector.push_back(theBadStripRange);

    SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(BadModule_, range))
      edm::LogError("SiStripBadChannelBuilder")
          << "[SiStripBadChannelBuilder::analyze] detid already exists" << std::endl;
  }
  //End now write sistripbadChannel data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripBadStripRcd")) {
      mydbservice->createOneIOV<SiStripBadStrip>(*obj, mydbservice->beginOfTime(), "SiStripBadStripRcd");
    } else {
      mydbservice->appendOneIOV<SiStripBadStrip>(*obj, mydbservice->currentTime(), "SiStripBadStripRcd");
    }
  } else {
    edm::LogError("SiStripBadStripBuilder") << "Service is unavailable" << std::endl;
  }

  return obj;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripBadChannelBuilder);
