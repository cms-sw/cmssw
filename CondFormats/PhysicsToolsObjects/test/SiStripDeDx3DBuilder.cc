#ifndef SiStripDeDx3DBuilder_H
#define SiStripDeDx3DBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
//#include "FWCore/Utilities/interface/FileInPath.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include <iostream>
#include <fstream>
//#include "CLHEP/Random/RandFlat.h"
//#include "CLHEP/Random/RandGauss.h"

class SiStripDeDx3DBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripDeDx3DBuilder(const edm::ParameterSet& iConfig);

  ~SiStripDeDx3DBuilder(){};

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  //edm::FileInPath fp_;
  bool printdebug_;
};

SiStripDeDx3DBuilder::SiStripDeDx3DBuilder(const edm::ParameterSet& iConfig) {}
//  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
//  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

void SiStripDeDx3DBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripDeDx3DBuilder")
      << "... creating dummy PhysicsToolsObjects::Calibration::HistogramD3D Data for Run " << run << "\n " << std::endl;

  //  PhysicsToolsObjects::Calibration::HistogramD2D* obj = new PhysicsTools::Calibration::HistogramD2D(300, 0., 3., 1000,0.,1000.);
  PhysicsTools::Calibration::HistogramD3D* obj =
      new PhysicsTools::Calibration::HistogramD3D(5, 0, 5, 100, 0., 3., 100, 0., 1000.);

  for (int ix = 0; ix < 5; ix++) {
    for (int iy = 0; iy < 100; iy++) {
      for (int iz = 0; iz < 100; iz++) {
        //        edm::LogInfo("SiStripDeDx3DBuilder") << "X = " << ix << " Y = " << iy << " Z = " << iz << std::endl;
        obj->setBinContent(ix, iy, iz, ix + 2 * iy + 3 * iz);
      }
    }
  }
  edm::LogInfo("SiStripDeDx3DBuilder") << "HISTO HAS BEEN FILLED" << std::endl;

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripDeDxProton_3D_Rcd")) {
      mydbservice->createNewIOV<PhysicsTools::Calibration::HistogramD3D>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripDeDxProton_3D_Rcd");
    } else {
      mydbservice->appendSinceTime<PhysicsTools::Calibration::HistogramD3D>(
          obj, mydbservice->currentTime(), "SiStripDeDxProton_3D_Rcd");
    }
  } else {
    edm::LogError("SiStripDeDx3DBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDx3DBuilder);

#endif
