#include "CondTools/SiStrip/plugins/SiStripThresholdBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripThresholdBuilder::SiStripThresholdBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 3)) {}

void SiStripThresholdBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripThresholdBuilder")
      << "... creating dummy SiStripThreshold Data for Run " << run << "\n " << std::endl;

  SiStripThreshold* obj = new SiStripThreshold();

  int count = -1;
  for (const auto& it : SiStripDetInfoFileReader::read(fp_.fullPath()).getAllData()) {
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

      obj->setData(strip, lTh, hTh, cTh, theSiStripVector);
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
      obj->setData(strip + 1, lTh, hTh, theSiStripVector);
      strip = (uint16_t)(CLHEP::RandFlat::shoot(strip + 2, 128 * it.second.nApvs));
    }
    if (!obj->put(it.first, theSiStripVector))
      edm::LogError("SiStripThresholdBuilder")
          << "[SiStripThresholdBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripThresholdRcd")) {
      mydbservice->createNewIOV<SiStripThreshold>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripThresholdRcd");
    } else {
      mydbservice->appendSinceTime<SiStripThreshold>(obj, mydbservice->currentTime(), "SiStripThresholdRcd");
    }
  } else {
    edm::LogError("SiStripThresholdBuilder") << "Service is unavailable" << std::endl;
  }
}
