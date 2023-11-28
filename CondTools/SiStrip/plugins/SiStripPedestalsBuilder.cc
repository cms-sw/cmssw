#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <iostream>
#include <fstream>

class SiStripPedestalsBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripPedestalsBuilder(const edm::ParameterSet& iConfig);

  ~SiStripPedestalsBuilder() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::FileInPath fp_;
  uint32_t printdebug_;
};

SiStripPedestalsBuilder::SiStripPedestalsBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripPedestalsBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripPedestalsBuilder")
      << "... creating dummy SiStripPedestals Data for Run " << run << "\n " << std::endl;

  SiStripPedestals obj;

  int count = -1;

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();

  for (const auto& it : DetInfos) {
    const auto& nAPVs = it.second.nApvs;

    count++;
    //Generate Pedestal for det detid
    SiStripPedestals::InputVector theSiStripVector;
    for (int strip = 0; strip < 128 * nAPVs; ++strip) {
      float MeanPed = 100;
      float RmsPed = 5;

      float ped = CLHEP::RandGauss::shoot(MeanPed, RmsPed);

      if (count < static_cast<int>(printdebug_))
        edm::LogInfo("SiStripPedestalsBuilder") << "detid " << it.first << " \t"
                                                << " strip " << strip << " \t" << ped << " \t" << std::endl;
      obj.setData(ped, theSiStripVector);
    }

    //SiStripPedestals::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if (!obj.put(it.first, theSiStripVector))
      edm::LogError("SiStripPedestalsBuilder")
          << "[SiStripPedestalsBuilder::analyze] detid already exists" << std::endl;
  }

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripPedestalsRcd")) {
      mydbservice->createOneIOV<SiStripPedestals>(obj, mydbservice->beginOfTime(), "SiStripPedestalsRcd");
    } else {
      mydbservice->appendOneIOV<SiStripPedestals>(obj, mydbservice->currentTime(), "SiStripPedestalsRcd");
    }
  } else {
    edm::LogError("SiStripPedestalsBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripPedestalsBuilder);
