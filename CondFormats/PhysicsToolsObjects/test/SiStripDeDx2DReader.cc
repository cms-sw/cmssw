#ifndef SiStripDeDx2DReader_H
#define SiStripDeDx2DReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_2D_Rcd.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

class SiStripDeDx2DReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripDeDx2DReader(const edm::ParameterSet&);
  ~SiStripDeDx2DReader();

  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  const edm::ESGetToken<PhysicsTools::Calibration::VHistogramD2D, SiStripDeDxProton_2D_Rcd> SiStripDeDx2DToken_;
};

SiStripDeDx2DReader::SiStripDeDx2DReader(const edm::ParameterSet& iConfig) : SiStripDeDx2DToken_(esConsumes()) {}

SiStripDeDx2DReader::~SiStripDeDx2DReader() = default;

void SiStripDeDx2DReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<PhysicsTools::Calibration::VHistogramD2D> SiStripDeDx2D_ = iSetup.getHandle(SiStripDeDx2DToken_);
  edm::LogInfo("SiStripDeDx2DReader") << "[SiStripDeDx2DReader::analyze] End Reading SiStripDeDxProton_2D" << std::endl;

  for (int ihisto = 0; ihisto < 3; ihisto++) {
    std::cout << (SiStripDeDx2D_->vHist)[ihisto].numberOfBinsX() << "   "
              << (SiStripDeDx2D_->vHist)[ihisto].numberOfBinsY() << std::endl;

    for (int ix = 0; ix < 300; ix++) {
      for (int iy = 0; iy < 1000; iy++) {
        std::cout << (SiStripDeDx2D_->vHist)[ihisto].binContent(ix, iy) << " "
                  << (SiStripDeDx2D_->vHist)[ihisto].value(ix / 100., iy) << std::endl;
      }
    }

    std::cout << "Value = " << (SiStripDeDx2D_->vValues)[ihisto] << std::endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDx2DReader);

#endif
