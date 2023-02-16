#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/PhysicsToolsObjects/test/SiStripDeDxMipBuilder.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <iostream>
#include <fstream>

SiStripDeDxMipBuilder::SiStripDeDxMipBuilder(const edm::ParameterSet& iConfig) {}
//  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
//  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

void SiStripDeDxMipBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripDeDxMipBuilder")
      << "... creating dummy PhysicsToolsObjects::Calibration::HistogramD2D Data for Run " << run << "\n " << std::endl;

  //  PhysicsToolsObjects::Calibration::HistogramD2D* obj = new PhysicsTools::Calibration::HistogramD2D(300, 0., 3., 1000,0.,1000.);
  PhysicsTools::Calibration::HistogramD2D obj(300, 0., 3., 1000, 0., 1000.);

  for (int ix = 0; ix < 300; ix++) {
    for (int iy = 0; iy < 1000; iy++) {
      obj.setBinContent(ix, iy, iy / 999.);
    }
  }

  //  SiStripDetInfoFileReader reader(fp_.fullPath());
  //  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();

  //  int count=-1;
  //  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){
  //   count++;
  //Generate Gain for det detid
  //  std::vector<float> theSiStripVector;
  //  for(unsigned short j=0; j<it->second.nApvs; j++){
  //   float gain= (j+1)*1000+ (RandFlat::shoot(1.)*100);
  //   if (count<printdebug_)
  //	edm::LogInfo("PhysicsToolsObjects::Calibration::HistogramD2DBuilder") << "detid " << it->first << " \t"
  // 					      << " apv " << j << " \t"
  // 					      << gain    << " \t"
  // 					      << std::endl;
  //       theSiStripVector.push_back(gain);
  //     }

  //     PhysicsToolsObjects::Calibration::HistogramD2D::Range range(theSiStripVector.begin(),theSiStripVector.end());
  //     if ( ! obj.put(it->first,range) )

  //      edm::LogError("PhysicsToolsObjects::Calibration::HistogramD2DBuilder")<<"[PhysicsToolsObjects::Calibration::HistogramD2DBuilder::analyze] detid already exists"<<std::endl;
  //  }

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripDeDxMipRcd")) {
      mydbservice->createOneIOV<PhysicsTools::Calibration::HistogramD2D>(
          obj, mydbservice->beginOfTime(), "SiStripDeDxMipRcd");
    } else {
      mydbservice->appendOneIOV<PhysicsTools::Calibration::HistogramD2D>(
          obj, mydbservice->currentTime(), "SiStripDeDxMipRcd");
    }
  } else {
    edm::LogError("SiStripDeDxMipBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDxMipBuilder);
