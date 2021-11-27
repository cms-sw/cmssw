// system include files
#include <cstdio>
#include <iostream>
#include <sys/time.h>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// ROOT includes
#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"

//
//
// class decleration
//
class SiPixelLorentzAngleReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelLorentzAngleReader(const edm::ParameterSet&);
  ~SiPixelLorentzAngleReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAToken_;
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleSimRcd> siPixelSimLAToken_;
  const bool printdebug_;
  const bool useSimRcd_;

  TH1F* LorentzAngleBarrel_;
  TH1F* LorentzAngleForward_;
};

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
    if (printdebug_) {
      edm::LogPrint("SiPixelLorentzAngleReader") << "detid " << it->first << " \t"
                                                 << " Lorentz angle  " << it->second << std::endl;
    }
    unsigned int subdet = DetId(it->first).subdetId();
    if (subdet == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      LorentzAngleBarrel_->Fill(it->second);
    } else if (subdet == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      LorentzAngleForward_->Fill(it->second);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
DEFINE_FWK_MODULE(SiPixelLorentzAngleReader);
