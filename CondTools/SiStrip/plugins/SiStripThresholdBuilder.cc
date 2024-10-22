#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <iostream>
#include <fstream>

class SiStripThresholdBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripThresholdBuilder(const edm::ParameterSet& iConfig);

  ~SiStripThresholdBuilder() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::FileInPath fp_;
  uint32_t printdebug_;
};

SiStripThresholdBuilder::SiStripThresholdBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 3)) {}

void SiStripThresholdBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripThresholdBuilder")
      << "... creating dummy SiStripThreshold Data for Run " << run << "\n " << std::endl;

  SiStripThreshold obj;

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();

  int count = -1;
  for (const auto& it : DetInfos) {
    count++;
    //Generate Pedestal for det detid
    SiStripThreshold::Container theSiStripVector;
    uint16_t strip = 0;
    while (strip < 128 * it.second.nApvs) {
      float lTh = (CLHEP::RandFlat::shoot(1.) * 64) / 5;
      float hTh = (CLHEP::RandFlat::shoot(1.) * 64) / 5;
      if (hTh < lTh) {
        float tmp = hTh;
        hTh = lTh;
        lTh = tmp;
      }
      float cTh = (CLHEP::RandFlat::shoot(1.) * 30.);

      obj.setData(strip, lTh, hTh, cTh, theSiStripVector);
      if (count < (int)printdebug_) {
        std::stringstream ss;
        theSiStripVector.back().print(ss);
        edm::LogInfo("SiStripThresholdBuilder")
            << "detid: " << it.first << " \n"
            << "firstStrip: " << strip << " \t"
            << "lTh: " << lTh << " \t"
            << "hTh: " << hTh << " \t"
            << "cTh: " << cTh << " \t"
            << "FirstStrip_and_Hth: " << theSiStripVector.back().FirstStrip_and_Hth << " \n"
            << ss.str() << std::endl;
      }
      obj.setData(strip + 1, lTh, hTh, theSiStripVector);
      strip = (uint16_t)(CLHEP::RandFlat::shoot(strip + 2, 128 * it.second.nApvs));
    }
    if (!obj.put(it.first, theSiStripVector))
      edm::LogError("SiStripThresholdBuilder")
          << "[SiStripThresholdBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripThresholdRcd")) {
      mydbservice->createOneIOV<SiStripThreshold>(obj, mydbservice->beginOfTime(), "SiStripThresholdRcd");
    } else {
      mydbservice->appendOneIOV<SiStripThreshold>(obj, mydbservice->currentTime(), "SiStripThresholdRcd");
    }
  } else {
    edm::LogError("SiStripThresholdBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripThresholdBuilder);
