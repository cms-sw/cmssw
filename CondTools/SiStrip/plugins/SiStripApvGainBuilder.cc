// system include files
#include <fstream>
#include <iostream>

// user include files
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

class SiStripApvGainBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripApvGainBuilder(const edm::ParameterSet& iConfig);

  ~SiStripApvGainBuilder() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::FileInPath fp_;
  const bool printdebug_;
};

SiStripApvGainBuilder::SiStripApvGainBuilder(const edm::ParameterSet& iConfig)
    : fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)) {}

void SiStripApvGainBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripApvGainBuilder") << "... creating dummy SiStripApvGain Data for Run " << run << "\n "
                                        << std::endl;

  SiStripApvGain obj;

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();

  int count = -1;
  for (const auto& it : DetInfos) {
    const auto& nAPVs = it.second.nApvs;
    count++;
    //Generate Gain for det detid
    std::vector<float> theSiStripVector;
    for (unsigned short j = 0; j < nAPVs; j++) {
      float gain = (j + 1) * 1000 + (CLHEP::RandFlat::shoot(1.) * 100);
      if (count < printdebug_)
        edm::LogInfo("SiStripApvGainBuilder") << "detid " << it.first << " \t"
                                              << " apv " << j << " \t" << gain << " \t" << std::endl;
      theSiStripVector.push_back(gain);
    }

    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj.put(it.first, range))
      edm::LogError("SiStripApvGainBuilder") << "[SiStripApvGainBuilder::analyze] detid already exists" << std::endl;
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
    edm::LogError("SiStripApvGainBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripApvGainBuilder);
