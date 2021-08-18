#ifndef SiStripDeDx2DBuilder_H
#define SiStripDeDx2DBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
//#include "FWCore/Utilities/interface/FileInPath.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"

#include <iostream>
#include <fstream>
//#include "CLHEP/Random/RandFlat.h"
//#include "CLHEP/Random/RandGauss.h"

class SiStripDeDx2DBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripDeDx2DBuilder(const edm::ParameterSet& iConfig);

  ~SiStripDeDx2DBuilder(){};

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  //edm::FileInPath fp_;
  bool printdebug_;
};

SiStripDeDx2DBuilder::SiStripDeDx2DBuilder(const edm::ParameterSet& iConfig) {}
//  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
//  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

void SiStripDeDx2DBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripDeDx2DBuilder")
      << "... creating dummy PhysicsToolsObjects::Calibration::HistogramD2D Data for Run " << run << "\n " << std::endl;

  //  PhysicsToolsObjects::Calibration::HistogramD2D* obj = new PhysicsTools::Calibration::HistogramD2D(300, 0., 3., 1000,0.,1000.);
  PhysicsTools::Calibration::VHistogramD2D* obj = new PhysicsTools::Calibration::VHistogramD2D();

  for (int ih = 0; ih < 3; ih++) {
    PhysicsTools::Calibration::HistogramD2D myhist(300, 0., 3., 1000, 0., 1000.);
    for (int ix = 0; ix < 300; ix++) {
      for (int iy = 0; iy < 1000; iy++) {
        myhist.setBinContent(ix, iy, iy / 999.);
      }
    }

    (obj->vHist).push_back(myhist);
    (obj->vValues).push_back(ih);
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
  //     if ( ! obj->put(it->first,range) )

  //      edm::LogError("PhysicsToolsObjects::Calibration::HistogramD2DBuilder")<<"[PhysicsToolsObjects::Calibration::HistogramD2DBuilder::analyze] detid already exists"<<std::endl;
  //  }

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripDeDxProton_2D_Rcd")) {
      mydbservice->createNewIOV<PhysicsTools::Calibration::VHistogramD2D>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripDeDxProton_2D_Rcd");
    } else {
      mydbservice->appendSinceTime<PhysicsTools::Calibration::VHistogramD2D>(
          obj, mydbservice->currentTime(), "SiStripDeDxProton_2D_Rcd");
    }
  } else {
    edm::LogError("SiStripDeDx2DBuilder") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDx2DBuilder);

#endif
