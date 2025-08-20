// system include files
#include <fstream>
#include <iostream>

// user include files
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/PhysicsToolsObjects/interface/DeDxCalibration.h"
#include "CondFormats/DataRecord/interface/DeDxCalibrationRcd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

class SiStripApvGainFromDeDxCalibration : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripApvGainFromDeDxCalibration(const edm::ParameterSet& iConfig);

  ~SiStripApvGainFromDeDxCalibration() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  typedef std::pair<uint32_t, unsigned char> ChipId;

private:
  const edm::ESGetToken<DeDxCalibration, DeDxCalibrationRcd> dedxCalibToken_;
  const edm::FileInPath fp_;
  const bool printdebug_;
};

SiStripApvGainFromDeDxCalibration::SiStripApvGainFromDeDxCalibration(const edm::ParameterSet& iConfig)
    : dedxCalibToken_(esConsumes<DeDxCalibration, DeDxCalibrationRcd>()),
      fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripApvGainFromDeDxCalibration::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripApvGainFromDeDxCalibration")
      << "... creating dummy SiStripApvGain Data for Run " << run << "\n " << std::endl;

  SiStripApvGain obj;

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();
  const auto& dedxCalib = iSetup.getData(dedxCalibToken_);

  int count = -1;
  for (const auto& it : DetInfos) {
    const auto& nAPVs = it.second.nApvs;
    count++;
    //Generate Gain for det detid
    std::vector<float> theSiStripVector;
    for (unsigned short j = 0; j < nAPVs; j++) {
      const auto& chipId = ChipId(it.first, j);
      const auto& g = dedxCalib.gain().find(chipId);

      if (g == dedxCalib.gain().end())
        continue;

      if (count < printdebug_)
        edm::LogInfo("SiStripApvGainFromDeDxCalibration") << "detid " << it.first << " \t"
                                                          << " apv " << j << " \t" << g->second << " \t" << std::endl;
      theSiStripVector.push_back(g->second);
    }

    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj.put(it.first, range))
      edm::LogError("SiStripApvGainFromDeDxCalibration")
          << "[SiStripApvGainFromDeDxCalibration::analyze] detid already exists" << std::endl;
  }

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripApvGainRcd")) {
      mydbservice->createOneIOV<SiStripApvGain>(obj, mydbservice->beginOfTime(), "SiStripApvGainRcd");
    } else {
      mydbservice->appendOneIOV<SiStripApvGain>(obj, mydbservice->currentTime(), "SiStripApvGainRcd");
    }
  } else {
    edm::LogError("SiStripApvGainFromDeDxCalibration") << "Service is unavailable" << std::endl;
  }
}

void SiStripApvGainFromDeDxCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile));
  desc.addUntracked<uint32_t>("printDebug", 1);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripApvGainFromDeDxCalibration);
