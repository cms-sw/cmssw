#include "CondTools/SiStrip/plugins/SiStripApvGainBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripApvGainBuilder::SiStripApvGainBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripApvGainBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripApvGainBuilder") << "... creating dummy SiStripApvGain Data for Run " << run << "\n "
                                        << std::endl;

  SiStripApvGain* obj = new SiStripApvGain();

  int count = -1;
  for (const auto& it : SiStripDetInfoFileReader::read(fp_.fullPath()).getAllData()) {
    count++;
    //Generate Gain for det detid
    std::vector<float> theSiStripVector;
    for (unsigned short j = 0; j < it.second.nApvs; j++) {
      float gain = (j + 1) * 1000 + (CLHEP::RandFlat::shoot(1.) * 100);
      if (count < printdebug_)
        edm::LogInfo("SiStripApvGainBuilder") << "detid " << it.first << " \t"
                                              << " apv " << j << " \t" << gain << " \t" << std::endl;
      theSiStripVector.push_back(gain);
    }

    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(it.first, range))
      edm::LogError("SiStripApvGainBuilder") << "[SiStripApvGainBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripApvGainRcd")) {
      mydbservice->createNewIOV<SiStripApvGain>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripApvGainRcd");
    } else {
      mydbservice->appendSinceTime<SiStripApvGain>(obj, mydbservice->currentTime(), "SiStripApvGainRcd");
    }
  } else {
    edm::LogError("SiStripApvGainBuilder") << "Service is unavailable" << std::endl;
  }
}
