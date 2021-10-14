#include "SiPixelLorentzAngleReader.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

using namespace cms;

SiPixelLorentzAngleReader::SiPixelLorentzAngleReader(const edm::ParameterSet& iConfig)
    : siPixelLAToken_(esConsumes()),
      siPixelSimLAToken_(esConsumes()),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      useSimRcd_(iConfig.getParameter<bool>("useSimRcd")) {
  usesResource(TFileService::kSharedResource);
}

SiPixelLorentzAngleReader::~SiPixelLorentzAngleReader() = default;

void SiPixelLorentzAngleReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const SiPixelLorentzAngle* SiPixelLorentzAngle_;
  if (useSimRcd_) {
    SiPixelLorentzAngle_ = &iSetup.getData(siPixelSimLAToken_);
  } else {
    SiPixelLorentzAngle_ = &iSetup.getData(siPixelLAToken_);
  }

  edm::LogInfo("SiPixelLorentzAngleReader")
      << "[SiPixelLorentzAngleReader::analyze] End Reading SiPixelLorentzAngle" << std::endl;
  edm::Service<TFileService> fs;
  LorentzAngleBarrel_ = fs->make<TH1F>("LorentzAngleBarrelPixel", "LorentzAngleBarrelPixel", 150, 0, 0.15);
  LorentzAngleForward_ = fs->make<TH1F>("LorentzAngleForwardPixel", "LorentzAngleForwardPixel", 150, 0, 0.15);
  std::map<unsigned int, float> detid_la = SiPixelLorentzAngle_->getLorentzAngles();
  std::map<unsigned int, float>::const_iterator it;
  for (it = detid_la.begin(); it != detid_la.end(); it++) {
    //	edm::LogPrint("SiPixelLorentzAngleReader")  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second  << std::endl;
    //edm::LogInfo("SiPixelLorentzAngleReader")  << "detid " << it->first << " \t" << " Lorentz angle  " << it->second;
    unsigned int subdet = DetId(it->first).subdetId();
    if (subdet == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      LorentzAngleBarrel_->Fill(it->second);
    } else if (subdet == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      LorentzAngleForward_->Fill(it->second);
    }
  }
}
DEFINE_FWK_MODULE(SiPixelLorentzAngleReader);
