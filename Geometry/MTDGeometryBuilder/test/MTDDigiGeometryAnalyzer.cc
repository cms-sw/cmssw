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
  void analyseRectangle(const GeomDetUnit& det);
  void checkRotation(const GeomDetUnit& det);
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
  edm::LogInfo("MTDDigiGeometryAnalyzer")
      << "Geometry node for MTDGeom is  " << &(*pDD) << "\n"
      << " # detectors = " << pDD->detUnits().size() << "\n"
      << " # types     = " << pDD->detTypes().size() << "\n"
      << " # BTL dets  = " << pDD->detsBTL().size() << "\n"
      << " # ETL dets  = " << pDD->detsETL().size() << "\n"
      << " # layers " << pDD->geomDetSubDetector(1) << "  = " << pDD->numberOfLayers(1) << "\n"
      << " # layers " << pDD->geomDetSubDetector(2) << "  = " << pDD->numberOfLayers(2) << "\n";
  sunitt << std::fixed << std::setw(7) << pDD->detUnits().size() << std::setw(7) << pDD->detTypes().size() << "\n";
  for (auto const& it : pDD->detUnits()) {
    if (dynamic_cast<const MTDGeomDetUnit*>((it)) != nullptr) {
      const BoundPlane& p = (dynamic_cast<const MTDGeomDetUnit*>((it)))->specificSurface();
      const MTDDetId mtdId(it->geographicalId());
      std::stringstream moduleLabel;
      if (mtdId.mtdSubDetector() == 1) {
        const BTLDetId btlId(it->geographicalId());
        moduleLabel << " BTL side " << btlId.mtdSide() << " Rod " << btlId.mtdRR() << " type/RU " << btlId.modType()
                    << "/" << btlId.runit() << " mod " << btlId.module();
      } else if (mtdId.mtdSubDetector() == 2) {
        const ETLDetId etlId(it->geographicalId());
        moduleLabel << " ETL side " << mtdId.mtdSide() << " Disc/Side/Sector " << etlId.nDisc() << " "
                    << etlId.discSide() << " " << etlId.sector();
      } else {
        edm::LogWarning("MTDDigiGeometryanalyzer") << (it->geographicalId()).rawId() << " unknown MTD subdetector!";
      }
      edm::LogVerbatim("MTDDigiGeometryAnalyzer")
          << "---------------------------------------------------------- \n"
          << it->geographicalId().rawId() << moduleLabel.str() << " RadLeng Pixel " << p.mediumProperties().radLen()
          << " Xi Pixel " << p.mediumProperties().xi();

      const GeomDetUnit theDet = *(dynamic_cast<const MTDGeomDetUnit*>(it));
      analyseRectangle(theDet);
    }
  }

  for (auto const& it : pDD->detTypes()) {
    if (dynamic_cast<const MTDGeomDetType*>((it)) != nullptr) {
      const PixelTopology& p = (dynamic_cast<const MTDGeomDetType*>((it)))->specificTopology();
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(p);
      auto pitchval = topo.pitch();
      edm::LogVerbatim("MTDDigiGeometryAnalyzer")
          << "\n Subdetector " << it->subDetector() << " MTD Det " << it->name() << "\n"
          << " Rows     " << topo.nrows() << " Columns " << topo.ncolumns() << " ROCS X   " << topo.rocsX()
          << " ROCS Y  " << topo.rocsY() << " Rows/ROC " << topo.rowsperroc() << " Cols/ROC " << topo.colsperroc()
          << " Pitch X " << pitchval.first << " Pitch Y " << pitchval.second << " Sensor Interpad X "
          << topo.gapxInterpad() << " Sensor Interpad Y " << topo.gapyInterpad() << " Sensor Border X "
          << topo.gapxBorder() << " Sensor Border Y " << topo.gapyBorder();
      sunitt << std::fixed << std::setw(7) << it->subDetector() << std::setw(4) << topo.nrows() << std::setw(4)
             << topo.ncolumns() << std::setw(4) << std::setw(4) << topo.rocsX() << std::setw(4) << topo.rocsY()
             << std::setw(4) << topo.rowsperroc() << std::setw(4) << topo.colsperroc() << std::setw(10)
             << pitchval.first << std::setw(10) << pitchval.second << std::setw(10) << topo.gapxInterpad()
             << std::setw(10) << topo.gapyInterpad() << std::setw(10) << topo.gapxBorder() << std::setw(10)
             << topo.gapyBorder() << "\n";
      checkRectangularMTDTopology(topo);
    }
  }

  edm::LogInfo("MTDDigiGeometryAnalyzer") << "Acceptance of BTL module:";
  auto const& btldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsBTL().front()));
  checkPixelsAcceptance(btldet);
  edm::LogInfo("MTDDigiGeometryAnalyzer") << "Acceptance of ETL module:";
  auto const& etldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsETL().front()));
  checkPixelsAcceptance(etldet);

  edm::LogInfo("MTDDigiGeometryAnalyzer") << "Additional MTD geometry content:"
                                          << "\n"
                                          << " # dets            = " << pDD->dets().size() << "\n"
                                          << " # detUnitIds      = " << pDD->detUnitIds().size() << "\n"
                                          << " # detIds          = " << pDD->detIds().size() << "\n";
  sunitt << std::fixed << std::setw(7) << pDD->dets().size() << std::setw(7) << pDD->detUnitIds().size() << std::setw(7)
         << pDD->detIds().size() << "\n";

  edm::LogVerbatim("MTDUnitTest") << sunitt.str();
}

void MTDDigiGeometryAnalyzer::checkRectangularMTDTopology(const RectangularMTDTopology& topo) {
  std::stringstream pixelinfo;
  pixelinfo << "Pixel center location:\n";
  LocalPoint center(0, 0, 0);
  for (int r = 0; r < topo.nrows(); r++) {
    for (int c = 0; c < topo.ncolumns(); c++) {
      sunitt << r << " " << c << " " << topo.pixelToModuleLocalPoint(center, r, c) << "\n";
      pixelinfo << r << " " << c << " " << topo.pixelToModuleLocalPoint(center, r, c) << "\n";
    }
  }
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << pixelinfo.str();
}

void MTDDigiGeometryAnalyzer::analyseRectangle(const GeomDetUnit& det) {
  const double safety = 0.9999;

  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  const GlobalPoint& pos = det.position();
  double length = tb->length();
  double width = tb->width();
  double thickness = tb->thickness();

  GlobalVector yShift = det.surface().toGlobal(LocalVector(0, 0, safety * length / 2.));
  GlobalPoint outerMiddle = pos + yShift;
  GlobalPoint innerMiddle = pos + (-1. * yShift);
  if (outerMiddle.perp() < innerMiddle.perp())
    std::swap(outerMiddle, innerMiddle);

  auto fround = [&](double in) {
    std::stringstream ss;
    ss << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  };

  auto fvecround = [&](GlobalPoint vecin) {
    std::stringstream ss;
    ss << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  };

  edm::LogVerbatim("MTDDigiGeometryAnalyzer")
      << "Det at pos " << fvecround(pos) << " radius " << fround(std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()))
      << " has length " << fround(length) << " width " << fround(width) << " thickness " << fround(thickness) << "\n"
      << "det center inside bounds? " << tb->inside(det.surface().toLocal(pos)) << "\n"
      << "outerMiddle " << fvecround(outerMiddle);
  sunitt << det.geographicalId().rawId() << fvecround(pos) << fround(length) << fround(width) << fround(thickness)
         << tb->inside(det.surface().toLocal(pos)) << fvecround(outerMiddle) << "\n";

  checkRotation(det);
}

void MTDDigiGeometryAnalyzer::checkRotation(const GeomDetUnit& det) {
  const double eps = std::numeric_limits<float>::epsilon();
  static int first = 0;
  if (first == 0) {
    edm::LogVerbatim("MTDDigiGeometryAnalyzer")
        << "numeric_limits<float>::epsilon() " << std::numeric_limits<float>::epsilon();
    first = 1;
  }

  const Surface::RotationType& rot(det.surface().rotation());
  GlobalVector a(rot.xx(), rot.xy(), rot.xz());
  GlobalVector b(rot.yx(), rot.yy(), rot.yz());
  GlobalVector c(rot.zx(), rot.zy(), rot.zz());
  GlobalVector cref = a.cross(b);
  GlobalVector aref = b.cross(c);
  GlobalVector bref = c.cross(a);
  if ((a - aref).mag() > eps || (b - bref).mag() > eps || (c - cref).mag() > eps) {
    edm::LogWarning("MTDDigiGeometryAnalyzer")
        << " Rotation not good by cross product: " << (a - aref).mag() << ", " << (b - bref).mag() << ", "
        << (c - cref).mag() << " for det at pos " << det.surface().position();
  }
  if (fabs(a.mag() - 1.) > eps || fabs(b.mag() - 1.) > eps || fabs(c.mag() - 1.) > eps) {
    edm::LogWarning("MTDDigiGeometryAnalyzer") << " Rotation not good by bector mag: " << (a).mag() << ", " << (b).mag()
                                               << ", " << (c).mag() << " for det at pos " << det.surface().position();
  }
}

void MTDDigiGeometryAnalyzer::checkPixelsAcceptance(const GeomDetUnit& det) {

  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  double length = tb->length();
  double width = tb->width();
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "X (width) = " << width << " Y (length) = " << length;

  const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det.topology());
  const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

  const size_t maxindex = 100000;
  size_t inpixel(0);
  for (size_t index = 0; index < maxindex; index++) {
    double ax = CLHEP::RandFlat::shoot(-width*0.5, width*0.5);
    double ay = CLHEP::RandFlat::shoot(-length*0.5, length*0.5);
    LocalPoint hit(ax, ay, 0);
    if (topo.isInPixel(hit)) {
      inpixel++;
    }
  }
  double acc = (double)inpixel/(double)maxindex;
  double accerr = std::sqrt(acc*(1.-acc)/(double)maxindex);
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "Acceptance: "<< acc << " +/- " << accerr;

}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiGeometryAnalyzer);
