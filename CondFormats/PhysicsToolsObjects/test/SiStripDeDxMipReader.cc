// system includes
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

// user include files
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMipRcd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripDeDxMipReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripDeDxMipReader(const edm::ParameterSet&);
  ~SiStripDeDxMipReader();

  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  const edm::ESGetToken<PhysicsTools::Calibration::HistogramD2D, SiStripDeDxMipRcd> SiStripDeDxMipToken_;
};

using namespace cms;

SiStripDeDxMipReader::SiStripDeDxMipReader(const edm::ParameterSet& iConfig) : SiStripDeDxMipToken_(esConsumes()) {}

SiStripDeDxMipReader::~SiStripDeDxMipReader() = default;

void SiStripDeDxMipReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<PhysicsTools::Calibration::HistogramD2D> SiStripDeDxMip_ = iSetup.getHandle(SiStripDeDxMipToken_);
  edm::LogInfo("SiStripDeDxMipReader") << "[SiStripDeDxMipReader::analyze] End Reading SiStripDeDxMip" << std::endl;
  std::cout << SiStripDeDxMip_->numberOfBinsX() << "   " << SiStripDeDxMip_->numberOfBinsY() << std::endl;
  for (int ix = 0; ix < 300; ix++) {
    for (int iy = 0; iy < 1000; iy++) {
      std::cout << SiStripDeDxMip_->binContent(ix, iy) << " " << SiStripDeDxMip_->value(ix / 100., iy) << std::endl;
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDxMipReader);
