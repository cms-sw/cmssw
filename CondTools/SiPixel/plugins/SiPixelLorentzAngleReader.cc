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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAToken_;
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleSimRcd> siPixelSimLAToken_;
  const std::string siPixelLALabel_;
  const std::string siPixelSimLALabel_;
  const uint32_t printdebug_;
  const bool useSimRcd_;
  TH1F* LorentzAngleBarrel_;
  TH1F* LorentzAngleForward_;
};

using namespace cms;

SiPixelLorentzAngleReader::SiPixelLorentzAngleReader(const edm::ParameterSet& iConfig)
    : siPixelLAToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("recoLabel")))),
      siPixelSimLAToken_(esConsumes((edm::ESInputTag("", iConfig.getParameter<std::string>("simLabel"))))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 10)),
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
  unsigned int count = 0;
  for (it = detid_la.begin(); it != detid_la.end(); it++) {
    count++;
    if (count <= printdebug_) {
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
  edm::LogPrint("SiPixelLorentzAngleReader")
      << "SiPixelLorentzAngleReader::" << __FUNCTION__ << "(...) :examined " << count << " DetIds" << std::endl;
}

void SiPixelLorentzAngleReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("EDAnalyzer to read per-module SiPixelLorentzAngle payloads in the EventSetup");
  desc.add<std::string>("recoLabel", "")->setComment("label for the reconstruction tags");
  desc.add<std::string>("simLabel", "")->setComment("label for the simulation tags");
  desc.addUntracked<unsigned int>("printDebug", 10);
  desc.add<bool>("useSimRcd", false);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
DEFINE_FWK_MODULE(SiPixelLorentzAngleReader);
