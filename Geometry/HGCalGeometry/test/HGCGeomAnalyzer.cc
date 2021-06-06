// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//STL headers
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "vdt/vdtMath.h"

using namespace std;

//
// class declaration
//
class HGCGeomAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HGCGeomAnalyzer(const edm::ParameterSet &);
  ~HGCGeomAnalyzer() override;

private:
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  const std::string txtFileName_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

  std::map<std::pair<DetId::Detector, int>, TProfile2D *> layerXYview_;
  std::map<std::pair<DetId::Detector, int>, TProfile *> layerThickR_;
  std::map<std::pair<DetId::Detector, int>, TProfile *> layerThickEta_;
};

//
HGCGeomAnalyzer::HGCGeomAnalyzer(const edm::ParameterSet &iConfig)
    : txtFileName_(iConfig.getParameter<std::string>("fileName")),
      geomToken_{esConsumes<CaloGeometry, CaloGeometryRecord>()} {
  usesResource("TFileService");
  fs_->file().cd();
}

//
HGCGeomAnalyzer::~HGCGeomAnalyzer() {}

//
void HGCGeomAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &es) {
  //output text file
  ofstream boundaries;
  boundaries.open(txtFileName_);

  //get geometry
  const CaloGeometry *caloGeom = &es.getData(geomToken_);

  std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
  for (const auto &d : dets) {
    const HGCalGeometry *geom =
        static_cast<const HGCalGeometry *>(caloGeom->getSubdetectorGeometry(d, ForwardSubdetector::ForwardEmpty));
    const HGCalTopology *topo = &(geom->topology());
    const HGCalDDDConstants *ddd = &(topo->dddConstants());

    //sub-detector boundaries
    unsigned int nlay = ddd->layers(true);
    unsigned int firstLay = ddd->firstLayer();

    const std::vector<DetId> &detIdVec = geom->getValidDetIds();
    std::map<int, std::map<HGCSiliconDetId::waferType, std::map<double, double>>> typeRadMap;

    //prepare header
    boundaries << "Subdetector: " << d << " has " << detIdVec.size() << " valid cells and " << nlay << " layers\n";
    boundaries << std::setw(5) << "layer" << std::setw(15) << "z" << std::setw(15) << "rIn" << std::setw(15) << "etaIn"
               << std::setw(15) << "rOut" << std::setw(15) << "etaOut" << std::setw(15);

    if (d != DetId::HGCalHSc) {
      boundaries << "rInThin" << std::setw(15) << "etaInThin" << std::setw(15) << "rInThick" << std::setw(15)
                 << "etaInThick" << std::setw(15);

      //book histograms
      TString baseName(Form("d%d_", d));
      TString title(d == DetId::HGCalEE ? "CEE" : "CEH_{Si}");

      std::vector<int> sides = {-1, 1};
      for (const auto &zside : sides) {
        for (unsigned int ilay = firstLay; ilay < firstLay + nlay; ++ilay) {
          //layer and side histos
          int signedLayer = ilay * zside;
          std::pair<DetId::Detector, int> key(d, signedLayer);
          TString layerBaseName(Form("%slayer%d_", baseName.Data(), signedLayer));
          TString layerTitle(Form("%s %d", title.Data(), signedLayer));
          layerXYview_[key] = fs_->make<TProfile2D>(
              layerBaseName + "xy_view", layerTitle + "; x [cm]; y [cm]; wafer type", 200, -200, 200, 200, -200, 200);
          layerThickR_[key] =
              fs_->make<TProfile>(layerBaseName + "thickness_vs_r", layerTitle + "; r [cm]; wafer type", 200, 0, 200);
          layerThickEta_[key] = fs_->make<TProfile>(
              layerBaseName + "thickness_vs_eta", layerTitle + "; abs(eta); wafer type", 200, 1.4, 3.3);
        }
      }

      //for Si loop over the detIds to find the thickness transitions
      for (const auto &cellId : detIdVec) {
        HGCSiliconDetId id(cellId.rawId());
        GlobalPoint pt = geom->getPosition(id);

        int layer = id.layer() * id.zside();
        double r(pt.perp());
        double eta(pt.eta());
        HGCSiliconDetId::waferType wt = (HGCSiliconDetId::waferType)id.type();

        //fill histograms
        std::pair<DetId::Detector, int> key(d, layer);
        layerXYview_[key]->Fill(pt.x(), pt.y(), (int)wt + 1);
        layerThickR_[key]->Fill(r, (int)wt + 1);
        layerThickEta_[key]->Fill(fabs(eta), (int)wt + 1);

        //fill map for boundary summary only for positive side
        if (pt.z() > 0)
          typeRadMap[layer][wt][r] = eta;
      }
    }
    boundaries << "\n";

    //loop over map and print transitions
    for (unsigned int ilay = firstLay; ilay < firstLay + nlay; ++ilay) {
      double zz = ddd->waferZ(ilay, true);
      auto rr = ddd->rangeR(zz, true);

      double rIn = rr.first;
      double rOut = rr.second;

      math::XYZVector rInVec(0., rIn, zz);
      math::XYZVector rOutVec(0., rOut, zz);
      boundaries << std::setw(5) << ilay << std::setw(15) << rInVec.z() << std::setw(15) << rInVec.y() << std::setw(15)
                 << rInVec.eta() << std::setw(15) << rOutVec.y() << std::setw(15) << rOutVec.eta() << std::setw(15);

      if (d != DetId::HGCalHSc) {
        boundaries << typeRadMap[ilay][HGCSiliconDetId::waferType::HGCalCoarseThin].begin()->first << std::setw(15)
                   << typeRadMap[ilay][HGCSiliconDetId::waferType::HGCalCoarseThin].begin()->second << std::setw(15)
                   << typeRadMap[ilay][HGCSiliconDetId::waferType::HGCalCoarseThick].begin()->first << std::setw(15)
                   << typeRadMap[ilay][HGCSiliconDetId::waferType::HGCalCoarseThick].begin()->second << std::setw(15);
      }
      boundaries << "\n";
    }

    boundaries << "\n";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeomAnalyzer);
