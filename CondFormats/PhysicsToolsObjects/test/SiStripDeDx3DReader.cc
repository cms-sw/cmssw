#ifndef SiStripDeDx3DReader_H
#define SiStripDeDx3DReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

class SiStripDeDx3DReader : public edm::EDAnalyzer {
public:
  explicit SiStripDeDx3DReader(const edm::ParameterSet&);
  ~SiStripDeDx3DReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  //  uint32_t printdebug_;
};

SiStripDeDx3DReader::SiStripDeDx3DReader(const edm::ParameterSet& iConfig) {}
//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

SiStripDeDx3DReader::~SiStripDeDx3DReader() {}

void SiStripDeDx3DReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<PhysicsTools::Calibration::HistogramD3D> SiStripDeDx3D_;
  iSetup.get<SiStripDeDxProton_3D_Rcd>().get(SiStripDeDx3D_);
  edm::LogInfo("SiStripDeDx3DReader") << "[SiStripDeDx3DReader::analyze] End Reading SiStripDeDxProton_3D" << std::endl;

  for (int ix = 0; ix < 5; ix++) {
    for (int iy = 0; iy < 100; iy++) {
      for (int iz = 0; iz < 100; iz++) {
        double Got = SiStripDeDx3D_->binContent(ix, iy, iz);
        std::cout << " X = " << ix << "Y = " << iy << " Z = " << iz << " --> " << Got << std::endl;
        if (Got != ix + 2 * iy + 3 * iz)
          std::cout << " BUG --> expected is " << ix + 2 * iy + 3 * iz << std::endl;
      }
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDx3DReader);

#endif
