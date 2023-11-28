// system includes
#include <iostream>
#include <fstream>

// user includes
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripNoisesBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripNoisesBuilder(const edm::ParameterSet& iConfig);

  ~SiStripNoisesBuilder() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::FileInPath fp_;
  const uint32_t printdebug_;
};

SiStripNoisesBuilder::SiStripNoisesBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripNoisesBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripNoisesBuilder") << "... creating dummy SiStripNoises Data for Run " << run << "\n " << std::endl;

  SiStripNoises obj;

  int count = -1;

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();

  for (const auto& it : DetInfos) {
    const auto& nAPVs = it.second.nApvs;

    count++;
    //Generate Noise for det detid
    SiStripNoises::InputVector theSiStripVector;
    for (int strip = 0; strip < 128 * nAPVs; ++strip) {
      float MeanNoise = 5;
      float RmsNoise = 1;
      float noise = CLHEP::RandGauss::shoot(MeanNoise, RmsNoise);

      //double badStripProb = .5;
      //bool disable = (CLHEP::RandFlat::shoot(1.) < badStripProb ? true:false);

      obj.setData(noise, theSiStripVector);
      if (count < static_cast<int>(printdebug_))
        edm::LogInfo("SiStripNoisesBuilder")
            << "detid " << it.first << " \t"
            << " strip " << strip << " \t" << noise << " \t" << theSiStripVector.back() / 10 << " \t" << std::endl;
    }
    if (!obj.put(it.first, theSiStripVector))
      edm::LogError("SiStripNoisesBuilder") << "[SiStripNoisesBuilder::analyze] detid already exists" << std::endl;
  }

  //And now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripNoisesRcd")) {
      mydbservice->createOneIOV<SiStripNoises>(obj, mydbservice->beginOfTime(), "SiStripNoisesRcd");
    } else {
      mydbservice->appendOneIOV<SiStripNoises>(obj, mydbservice->currentTime(), "SiStripNoisesRcd");
    }
  } else {
    edm::LogError("SiStripNoisesBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripNoisesBuilder);
