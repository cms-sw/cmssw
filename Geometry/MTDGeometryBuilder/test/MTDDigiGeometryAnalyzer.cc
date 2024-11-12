// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/Rounding.h"

#include <fstream>

#include "CLHEP/Random/RandFlat.h"

// class declaration

class MTDDigiGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MTDDigiGeometryAnalyzer(const edm::ParameterSet&);
  ~MTDDigiGeometryAnalyzer() override = default;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void checkRectangularMTDTopology(const RectangularMTDTopology&);
  void checkPixelsAcceptance(const GeomDetUnit& det);

  std::stringstream sunitt;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
};

using cms_rounding::roundIfNear0, cms_rounding::roundVecIfNear0;

MTDDigiGeometryAnalyzer::MTDDigiGeometryAnalyzer(const edm::ParameterSet& iConfig) {
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
}

// ------------ method called to produce the data  ------------
void MTDDigiGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //
  // get the MTDGeometry
  //
  auto pDD = iSetup.getTransientHandle(mtdgeoToken_);
  sunitt << "MTDGeometry:\n"
         << " # detectors  = " << pDD->detUnits().size() << "\n"
         << " # types      = " << pDD->detTypes().size() << "\n"
         << " # BTL dets   = " << pDD->detsBTL().size() << "\n"
         << " # ETL dets   = " << pDD->detsETL().size() << "\n"
         << " # layers " << pDD->geomDetSubDetector(1) << "   = " << pDD->numberOfLayers(1) << "\n"
         << " # layers " << pDD->geomDetSubDetector(2) << "   = " << pDD->numberOfLayers(2) << "\n"
         << " # dets       = " << pDD->dets().size() << "\n"
         << " # detUnitIds = " << pDD->detUnitIds().size() << "\n"
         << " # detIds     = " << pDD->detIds().size() << "\n";

  for (auto const& it : pDD->detTypes()) {
    if (dynamic_cast<const MTDGeomDetType*>((it)) != nullptr) {
      const PixelTopology& p = (dynamic_cast<const MTDGeomDetType*>((it)))->specificTopology();
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(p);
      auto pitchval = topo.pitch();
      sunitt << "\n Subdetector " << it->subDetector() << " MTD Det " << it->name() << "\n"
             << " Rows     " << topo.nrows() << " Columns " << topo.ncolumns() << " ROCS X   " << topo.rocsX()
             << " ROCS Y  " << topo.rocsY() << " Rows/ROC " << topo.rowsperroc() << " Cols/ROC " << topo.colsperroc()
             << " Pitch X " << pitchval.first << " Pitch Y " << pitchval.second << " Sensor Interpad X "
             << topo.gapxInterpad() << " Sensor Interpad Y " << topo.gapyInterpad() << " Sensor Border X "
             << topo.gapxBorder() << " Sensor Border Y " << topo.gapyBorder() << "\n";
      checkRectangularMTDTopology(topo);
    }
  }

  sunitt << "\nAcceptance of BTL module:";
  auto const& btldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsBTL().front()));
  checkPixelsAcceptance(btldet);
  sunitt << "\nAcceptance of ETL module:";
  auto const& etldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsETL().front()));
  checkPixelsAcceptance(etldet);

  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << sunitt.str();
  edm::LogVerbatim("MTDUnitTest") << sunitt.str();
}

void MTDDigiGeometryAnalyzer::checkRectangularMTDTopology(const RectangularMTDTopology& topo) {
  sunitt << "Pixel center location:\n";
  LocalPoint center(0, 0, 0);
  for (int r = 0; r < topo.nrows(); r++) {
    for (int c = 0; c < topo.ncolumns(); c++) {
      sunitt << std::setw(7) << r << std::setw(7) << c << " " << topo.pixelToModuleLocalPoint(center, r, c) << "\n";
    }
  }
}

void MTDDigiGeometryAnalyzer::checkPixelsAcceptance(const GeomDetUnit& det) {
  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  double length = tb->length();
  double width = tb->width();
  sunitt << " X (width) = " << width << " Y (length) = " << length;

  const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det.topology());
  const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

  const size_t maxindex = 100000;
  size_t inpixel(0);
  for (size_t index = 0; index < maxindex; index++) {
    double ax = CLHEP::RandFlat::shoot(-width * 0.5, width * 0.5);
    double ay = CLHEP::RandFlat::shoot(-length * 0.5, length * 0.5);
    LocalPoint hit(ax, ay, 0);
    auto const indici = topo.pixelIndex(hit);
    assert(indici.first < topo.nrows() && indici.second < topo.ncolumns());  // sanity check on the index definition
    if (topo.isInPixel(hit)) {
      inpixel++;
    }
  }
  double acc = (double)inpixel / (double)maxindex;
  double accerr = std::sqrt(acc * (1. - acc) / (double)maxindex);
  sunitt << " Acceptance: " << acc << " +/- " << accerr;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiGeometryAnalyzer);
