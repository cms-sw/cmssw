#include "CondTools/SiStrip/plugins/SiStripNoisesBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripNoisesBuilder::SiStripNoisesBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripNoisesBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripNoisesBuilder") << "... creating dummy SiStripNoises Data for Run " << run << "\n " << std::endl;

  SiStripNoises* obj = new SiStripNoises();

  int count = -1;
  for (const auto& it : SiStripDetInfoFileReader::read(fp_.fullPath()).getAllData()) {
    count++;
    //Generate Noise for det detid
    SiStripNoises::InputVector theSiStripVector;
    for (int strip = 0; strip < 128 * it.second.nApvs; ++strip) {
      float MeanNoise = 5;
      float RmsNoise = 1;
      float noise = CLHEP::RandGauss::shoot(MeanNoise, RmsNoise);

      //double badStripProb = .5;
      //bool disable = (CLHEP::RandFlat::shoot(1.) < badStripProb ? true:false);

      obj->setData(noise, theSiStripVector);
      if (count < static_cast<int>(printdebug_))
        edm::LogInfo("SiStripNoisesBuilder")
            << "detid " << it.first << " \t"
            << " strip " << strip << " \t" << noise << " \t" << theSiStripVector.back() / 10 << " \t" << std::endl;
    }

    if (!obj->put(it.first, theSiStripVector))
      edm::LogError("SiStripNoisesBuilder") << "[SiStripNoisesBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripNoisesRcd")) {
      mydbservice->createNewIOV<SiStripNoises>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripNoisesRcd");
    } else {
      //mydbservice->createNewIOV<SiStripNoises>(obj,mydbservice->currentTime(),"SiStripNoisesRcd");
      mydbservice->appendSinceTime<SiStripNoises>(obj, mydbservice->currentTime(), "SiStripNoisesRcd");
    }
  } else {
    edm::LogError("SiStripNoisesBuilder") << "Service is unavailable" << std::endl;
  }
}
