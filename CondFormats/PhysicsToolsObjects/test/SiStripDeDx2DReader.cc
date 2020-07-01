#ifndef SiStripDeDx2DReader_H
#define SiStripDeDx2DReader_H

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
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_2D_Rcd.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

class SiStripDeDx2DReader : public edm::EDAnalyzer {
public:
  explicit SiStripDeDx2DReader(const edm::ParameterSet&);
  ~SiStripDeDx2DReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  //  uint32_t printdebug_;
};

SiStripDeDx2DReader::SiStripDeDx2DReader(const edm::ParameterSet& iConfig) {}
//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

SiStripDeDx2DReader::~SiStripDeDx2DReader() {}

void SiStripDeDx2DReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<PhysicsTools::Calibration::VHistogramD2D> SiStripDeDx2D_;
  iSetup.get<SiStripDeDxProton_2D_Rcd>().get(SiStripDeDx2D_);
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

  //   std::vector<uint32_t> detid;
  //   SiStripDeDx2D_->getDetIds(detid);
  //   edm::LogInfo("Number of detids ")  << detid.size() << std::endl;

  //   if (printdebug_)
  //     for (size_t id=0;id<detid.size() && id<printdebug_;id++)
  //       {
  // 	SiStripDeDx2D::Range range=SiStripDeDx2D_->getRange(detid[id]);

  // 	int apv=0;
  // 	for(int it=0;it<range.second-range.first;it++){
  // 	  edm::LogInfo("SiStripDeDx2DReader")  << "detid " << detid[id] << " \t"
  // 					     << " apv " << apv++ << " \t"
  // 					     << SiStripDeDx2D_->getDeDx2D(it,range)     << " \t"
  // 					     << std::endl;
  // 	}
  //       }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDx2DReader);

#endif
