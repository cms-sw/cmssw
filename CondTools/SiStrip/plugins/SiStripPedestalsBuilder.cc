#include "CondTools/SiStrip/plugins/SiStripPedestalsBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripPedestalsBuilder::SiStripPedestalsBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripPedestalsBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripPedestalsBuilder")
      << "... creating dummy SiStripPedestals Data for Run " << run << "\n " << std::endl;

  SiStripPedestals* obj = new SiStripPedestals();

  int count = -1;
  for (const auto& it : SiStripDetInfoFileReader::read(fp_.fullPath()).getAllData()) {
    count++;
    //Generate Pedestal for det detid
    SiStripPedestals::InputVector theSiStripVector;
    for (int strip = 0; strip < 128 * it.second.nApvs; ++strip) {
      float MeanPed = 100;
      float RmsPed = 5;

      float ped = CLHEP::RandGauss::shoot(MeanPed, RmsPed);

      if (count < static_cast<int>(printdebug_))
        edm::LogInfo("SiStripPedestalsBuilder") << "detid " << it.first << " \t"
                                                << " strip " << strip << " \t" << ped << " \t" << std::endl;
      obj->setData(ped, theSiStripVector);
    }

    //SiStripPedestals::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if (!obj->put(it.first, theSiStripVector))
      edm::LogError("SiStripPedestalsBuilder")
          << "[SiStripPedestalsBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripPedestalsRcd")) {
      mydbservice->createNewIOV<SiStripPedestals>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripPedestalsRcd");
    } else {
      //mydbservice->createNewIOV<SiStripPedestals>(obj,mydbservice->currentTime(),"SiStripPedestalsRcd");
      mydbservice->appendSinceTime<SiStripPedestals>(obj, mydbservice->currentTime(), "SiStripPedestalsRcd");
    }
  } else {
    edm::LogError("SiStripPedestalsBuilder") << "Service is unavailable" << std::endl;
  }
}
