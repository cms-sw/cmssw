#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "TH1D.h"
#include "TH2D.h"

class HGCalGeometryCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalGeometryCheck(const edm::ParameterSet&);
  ~HGCalGeometryCheck() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  void doTest(const HGCalGeometry* geom, ForwardSubdetector subdet);
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det);

  const std::vector<std::string> nameDetectors_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> geomTokens_;
  const double rmin_, rmax_, zmin_, zmax_;
  const int nbinR_, nbinZ_;
  const bool ifNose_, verbose_;
  std::vector<unsigned int> modHF_;
  std::vector<TH2D*> h_RZ_;
  std::vector<TH1D*> h_Mod_;
};

HGCalGeometryCheck::HGCalGeometryCheck(const edm::ParameterSet& iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("detectorNames")),
      geomTokens_{edm::vector_transform(
          nameDetectors_,
          [this](const std::string& name) {
            return esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
          })},
      rmin_(iC.getUntrackedParameter<double>("rMin", 0.0)),
      rmax_(iC.getUntrackedParameter<double>("rMax", 300.0)),
      zmin_(iC.getUntrackedParameter<double>("zMin", 300.0)),
      zmax_(iC.getUntrackedParameter<double>("zMax", 600.0)),
      nbinR_(iC.getUntrackedParameter<int>("nBinR", 300)),
      nbinZ_(iC.getUntrackedParameter<int>("nBinZ", 600)),
      ifNose_(iC.getUntrackedParameter<bool>("ifNose", false)),
      verbose_(iC.getUntrackedParameter<bool>("verbosity", false)) {
  usesResource(TFileService::kSharedResource);
}

HGCalGeometryCheck::~HGCalGeometryCheck() {}

void HGCalGeometryCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  desc.add<std::vector<std::string>>("detectorNames", names);
  desc.addUntracked<double>("rMin", 0.0);
  desc.addUntracked<double>("rMax", 300.0);
  desc.addUntracked<double>("zMin", 300.0);
  desc.addUntracked<double>("zMax", 600.0);
  desc.addUntracked<int>("nBinR", 300);
  desc.addUntracked<int>("nBinZ", 600);
  desc.addUntracked<bool>("ifNose", false);
  desc.addUntracked<bool>("verbosity", false);
  descriptions.add("hgcalGeometryCheck", desc);
}

void HGCalGeometryCheck::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  edm::Service<TFileService> fs;
  char name[100], title[200];
  sprintf(name, "RZ_All");
  sprintf(title, "R vs Z for All Detectors");
  h_RZ_.emplace_back(fs->make<TH2D>(name, title, nbinZ_, zmin_, zmax_, nbinR_, rmin_, rmax_));

  for (unsigned int ih = 0; ih < nameDetectors_.size(); ++ih) {
    const auto& geomR = iSetup.getData(geomTokens_[ih]);
    const HGCalGeometry* geom = &geomR;
    int layerF = geom->topology().dddConstants().firstLayer();
    int layerL = geom->topology().dddConstants().lastLayer(true);
    edm::LogVerbatim("HGCalGeom") << nameDetectors_[ih] << " with layers in the range " << layerF << ":" << layerL;
    sprintf(name, "RZ_%s", nameDetectors_[ih].c_str());
    sprintf(title, "R vs Z for %s", nameDetectors_[ih].c_str());
    h_RZ_.emplace_back(fs->make<TH2D>(name, title, nbinZ_, zmin_, zmax_, nbinR_, rmin_, rmax_));
    unsigned int k(0);
    for (int lay = layerF; lay <= layerL; ++lay, ++k) {
      sprintf(name, "Mod_%s_L%d", nameDetectors_[ih].c_str(), lay);
      sprintf(title, "Modules in layer %d in %s", lay, nameDetectors_[ih].c_str());
      h_Mod_.emplace_back(fs->make<TH1D>(name, title, 200, -50, 50));

      auto zz = geom->topology().dddConstants().waferZ(lay, true);
      auto rr = geom->topology().dddConstants().rangeR(zz, true);
      auto rr0 = geom->topology().dddConstants().rangeRLayer(lay, true);
      edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " R " << rr.first << ":" << rr.second << " (" << rr0.first
                                    << ":" << rr0.second << ") Z " << zz;
      double r = rr.first;
      while (r <= rr.second) {
        h_RZ_[0]->Fill(zz, r);
        h_RZ_[ih + 1]->Fill(zz, r);
        for (int k = 0; k < 100; ++k) {
          double phi = 2 * k * M_PI / 100.0;
          GlobalPoint global1(r * cos(phi), r * sin(phi), zz);
          DetId id = geom->getClosestCell(global1);

          if (ifNose_) {
            HFNoseDetId detId = HFNoseDetId(id);
            h_Mod_.back()->Fill(detId.waferU());
            h_Mod_.back()->Fill(detId.waferV());
            if (verbose_)
              edm::LogVerbatim("HGCalGeom") << "R: " << r << " ID " << detId;
          } else if (geom->topology().waferHexagon6()) {
            HGCalDetId detId = HGCalDetId(id);
            h_Mod_.back()->Fill(detId.wafer());
            if (verbose_)
              edm::LogVerbatim("HGCalGeom") << "R: " << r << " ID " << detId;
          } else if (geom->topology().tileTrapezoid()) {
            HGCScintillatorDetId detId = HGCScintillatorDetId(id);
            h_Mod_.back()->Fill(detId.ieta());
            if (verbose_)
              edm::LogVerbatim("HGCalGeom") << "R: " << r << " ID " << detId;
          } else {
            HGCSiliconDetId detId = HGCSiliconDetId(id);
            h_Mod_.back()->Fill(detId.waferU());
            h_Mod_.back()->Fill(detId.waferV());
            if (verbose_)
              edm::LogVerbatim("HGCalGeom") << "R: " << r << " ID " << detId;
          }
        }
        r += 1.0;
      }
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalGeometryCheck);
