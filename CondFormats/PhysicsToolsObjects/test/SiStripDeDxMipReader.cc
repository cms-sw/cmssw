#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMipRcd.h"

#include "CondFormats/PhysicsToolsObjects/test/SiStripDeDxMipReader.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

using namespace cms;

SiStripDeDxMipReader::SiStripDeDxMipReader(const edm::ParameterSet& iConfig) {}
//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

SiStripDeDxMipReader::~SiStripDeDxMipReader() {}

void SiStripDeDxMipReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<PhysicsTools::Calibration::HistogramD2D> SiStripDeDxMip_;
  iSetup.get<SiStripDeDxMipRcd>().get(SiStripDeDxMip_);
  edm::LogInfo("SiStripDeDxMipReader") << "[SiStripDeDxMipReader::analyze] End Reading SiStripDeDxMip" << std::endl;
  std::cout << SiStripDeDxMip_->numberOfBinsX() << "   " << SiStripDeDxMip_->numberOfBinsY() << std::endl;
  for (int ix = 0; ix < 300; ix++) {
    for (int iy = 0; iy < 1000; iy++) {
      std::cout << SiStripDeDxMip_->binContent(ix, iy) << " " << SiStripDeDxMip_->value(ix / 100., iy) << std::endl;
    }
  }

  //   std::vector<uint32_t> detid;
  //   SiStripDeDxMip_->getDetIds(detid);
  //   edm::LogInfo("Number of detids ")  << detid.size() << std::endl;

  //   if (printdebug_)
  //     for (size_t id=0;id<detid.size() && id<printdebug_;id++)
  //       {
  // 	SiStripDeDxMip::Range range=SiStripDeDxMip_->getRange(detid[id]);

  // 	int apv=0;
  // 	for(int it=0;it<range.second-range.first;it++){
  // 	  edm::LogInfo("SiStripDeDxMipReader")  << "detid " << detid[id] << " \t"
  // 					     << " apv " << apv++ << " \t"
  // 					     << SiStripDeDxMip_->getDeDxMip(it,range)     << " \t"
  // 					     << std::endl;
  // 	}
  //       }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDeDxMipReader);
