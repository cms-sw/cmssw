#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalCornerCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalCornerCheck(const edm::ParameterSet&);
  ~HGCalCornerCheck() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string name_;
  const int32_t layerFirst_, layerLast_;
  const uint32_t nmax_, debug_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
};

HGCalCornerCheck::HGCalCornerCheck(const edm::ParameterSet& iC)
    : name_{iC.getParameter<std::string>("Detector")},
      layerFirst_{iC.getParameter<int32_t>("LayerFirst")},
      layerLast_{iC.getParameter<int32_t>("LayerLast")},
      nmax_{iC.getParameter<uint32_t>("NMax")},
      debug_{iC.getParameter<uint32_t>("Debug")},
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name_})},
      dets_(DetId::HGCalHSc) {
  edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name_ << " Layers " << layerFirst_ << ":" << layerLast_ << " NMax " << nmax_;
}

void HGCalCornerCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("Detector", "HGCalHEScintillatorSensitive");
  desc.add<int32_t>("LayerFirst", 8);
  desc.add<int32_t>("LayerLast", 21);
  desc.add<uint32_t>("NMax", 2);
  desc.add<uint32_t>("Debug", 0);
  descriptions.add("hgcalCornerCheck", desc);
}

void HGCalCornerCheck::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << name_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    const std::vector<DetId>& ids = geom->getValidDetIds();
    edm::LogVerbatim("HGCalGeomX") << "Test: " << ids.size() << " valid ids for " << name_;

    for (int layer = layerFirst_; layer <= layerLast_; ++layer) {
      unsigned int kk = 0;
      edm::LogVerbatim("HGCalGeomX") << "\nLayer: " << layer << "\n==========";
      for (DetId id : ids) {
	if ((HGCScintillatorDetId(id).layer() == layer) && (HGCScintillatorDetId(id).zside() > 0) && ((kk < nmax_) || (nmax_ <= 0))) {
	  ++kk;
	  std::vector<GlobalPoint> cor1 = geom->getCorners(id);
	  std::vector<GlobalPoint> cor2 = geom->get8Corners(id);
//        std::vector<GlobalPoint> cor3 = geom->getNewCorners(id, false);
	  std::ostringstream st1;
	  st1 << "Layer " << layer << " Ring " << HGCScintillatorDetId(id).ring() << " phi " << HGCScintillatorDetId(id).iphi() << " Corners:";
	  for (unsigned int k = 0; k < cor1.size(); ++k)
	    st1 << " " << cor1[k].perp();
	  st1 << " 8Corner8";
	  for (unsigned int k = 0; k < cor2.size(); ++k)
	    st1 << " " << cor2[k].perp();
//        st1 << " Newcorners:";
//        for (unsigned int k = 0; k < cor3.size(); ++k)
//          st1 << " " << cor3[k].perp();
	  edm::LogVerbatim("HGCalGeomX") << "Tile " << kk << " " << st1.str();
	  if (debug_ > 0) {
	    for (unsigned int k = 0; k < cor1.size(); ++k) {
	      std::ostringstream st2;
	      st2 << "  Corner[" << k << "]: (" << cor1[k].x() << ", " << cor1[k].y() << ", " << cor1[k].z() << ") perp " << cor1[k].perp();
	      edm::LogVerbatim("HGCalGeomX") << st2.str();
	    }
	    edm::LogVerbatim("HGCalGeomX") << "================================================================";
	  }
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalCornerCheck);
