// -*- C++ -*-
//
// Package:    TrackerDigiGeometryAnalyzer
// Class:      TrackerDigiGeometryAnalyzer
//
/**\class TrackerDigiGeometryAnalyzer TrackerDigiGeometryAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filippo Ambroglini
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
//
// class decleration
//

// #define PRINT(X) edm::LogInfo(X)
#define PRINT(X) std::cout << X << ':'

class TrackerDigiGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerDigiGeometryAnalyzer(const edm::ParameterSet&);
  ~TrackerDigiGeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void analyseTrapezoidal(const GeomDetUnit& det);
  void checkRotation(const GeomDetUnit& det);
  void checkTopology(const GeomDetUnit& det);
  std::ostream& cylindrical(std::ostream& os, const GlobalPoint& gp) const;
};

TrackerDigiGeometryAnalyzer::TrackerDigiGeometryAnalyzer(const edm::ParameterSet& iConfig) {}

TrackerDigiGeometryAnalyzer::~TrackerDigiGeometryAnalyzer() {}

// ------------ method called to produce the data  ------------
void TrackerDigiGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  PRINT("TrackerDigiGeometryAnalyzer") << "Here I am";
  //
  // get the TrackerGeom
  //
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  PRINT("TrackerDigiGeometryAnalyzer") << " Geometry node for TrackerGeom is  " << &(*pDD) << '\n';
  PRINT("TrackerDigiGeometryAnalyzer") << " I have " << pDD->detUnits().size() << " detectors" << '\n';
  PRINT("TrackerDigiGeometryAnalyzer") << " I have " << pDD->detTypes().size() << " types" << '\n';

  for (auto const& it : pDD->detUnits()) {
    if (dynamic_cast<const PixelGeomDetUnit*>((it)) != nullptr) {
      const BoundPlane& p = (dynamic_cast<const PixelGeomDetUnit*>((it)))->specificSurface();
      PRINT("TrackerDigiGeometryAnalyzer") << it->geographicalId() << " RadLeng Pixel " << p.mediumProperties().radLen()
                                           << ' ' << " Xi Pixel " << p.mediumProperties().xi() << '\n';
    }

    if (dynamic_cast<const StripGeomDetUnit*>((it)) != nullptr) {
      const BoundPlane& s = (dynamic_cast<const StripGeomDetUnit*>((it)))->specificSurface();
      PRINT("TrackerDigiGeometryAnalyzer") << it->geographicalId() << " RadLeng Strip " << s.mediumProperties().radLen()
                                           << " Xi Strip " << s.mediumProperties().xi() << '\n';
    }

    //analyseTrapezoidal(*it);
  }

  for (auto const& it : pDD->detTypes()) {
    if (dynamic_cast<const PixelGeomDetType*>((it)) != nullptr) {
      const PixelTopology& p = (dynamic_cast<const PixelGeomDetType*>((it)))->specificTopology();
      PRINT("TrackerDigiGeometryAnalyzer") << " PIXEL Det "  // << it->geographicalId()
                                           << "    Rows    " << p.nrows() << "    Columns " << p.ncolumns() << '\n';
    } else {
      const StripTopology& p = (dynamic_cast<const StripGeomDetType*>((it)))->specificTopology();
      PRINT("TrackerDigiGeometryAnalyzer") << " STRIP Det "  // << it->geographicalId()
                                           << "    Strips    " << p.nstrips() << '\n';
    }
  }
}
void TrackerDigiGeometryAnalyzer::analyseTrapezoidal(const GeomDetUnit& det) {
  // checkRotation( det);

  const double safety = 0.9999;

  const Bounds& bounds = det.surface().bounds();
  const TrapezoidalPlaneBounds* tb = dynamic_cast<const TrapezoidalPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  checkTopology(det);

  const GlobalPoint& pos = det.position();
  double length = tb->length();
  double width = tb->width();

  const std::array<const float, 4>& par = tb->parameters();
  double top = std::max(par[1], par[0]);
  double bot = std::min(par[1], par[0]);

  std::cout << std::endl;
  std::cout << "Det at pos " << pos << " has length " << length << " width " << width << " pars ";
  for (int i = 0; i < 4; i++)
    std::cout << par[i] << ", ";
  std::cout << std::endl;

  std::cout << "det center inside bounds? " << tb->inside(det.surface().toLocal(pos)) << std::endl;

  //   double outerScale = (pos.perp()+safety*length/2.) / pos.perp();
  //   GlobalPoint outerMiddle = GlobalPoint( outerScale*pos.x(), outerScale*pos.y(), pos.z());

  GlobalVector yShift = det.surface().toGlobal(LocalVector(0, safety * length / 2., 0));
  GlobalPoint outerMiddle = pos + yShift;
  GlobalPoint innerMiddle = pos + (-1. * yShift);
  if (outerMiddle.perp() < innerMiddle.perp())
    std::swap(outerMiddle, innerMiddle);

  GlobalVector upperShift = det.surface().toGlobal(LocalVector(safety * top, 0, 0));

  GlobalPoint ulc = outerMiddle + upperShift;
  GlobalPoint urc = outerMiddle + (-1. * upperShift);
  std::cout << "outerMiddle " << outerMiddle << " upperShift " << upperShift << " ulc "
            << "(" << ulc.perp() << "," << ulc.phi() << "," << ulc.z() << " urc "
            << "(" << urc.perp() << "," << urc.phi() << "," << urc.z() << std::endl;

  std::cout << "upper corners inside bounds? " << tb->inside(det.surface().toLocal(ulc)) << " "
            << tb->inside(det.surface().toLocal(urc)) << std::endl;

  //   double innerScale = (pos.perp()-safety*length/2.) / pos.perp();
  //   GlobalPoint innerMiddle = GlobalPoint( innerScale*pos.x(), innerScale*pos.y(), pos.z());

  GlobalVector lowerShift = det.surface().toGlobal(LocalVector(safety * bot, 0, 0));

  std::cout << "lower corners inside bounds? " << tb->inside(det.surface().toLocal(innerMiddle + lowerShift)) << " "
            << tb->inside(det.surface().toLocal(innerMiddle + (-1. * lowerShift))) << std::endl;
}

void TrackerDigiGeometryAnalyzer::checkRotation(const GeomDetUnit& det) {
  const double eps = std::numeric_limits<float>::epsilon();
  static int first = 0;
  if (first == 0) {
    std::cout << "numeric_limits<float>::epsilon() " << std::numeric_limits<float>::epsilon() << std::endl;
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
    std::cout << " Rotation not good by cross product: " << (a - aref).mag() << ", " << (b - bref).mag() << ", "
              << (c - cref).mag() << " for det at pos " << det.surface().position() << std::endl;
  }
  if (fabs(a.mag() - 1.) > eps || fabs(b.mag() - 1.) > eps || fabs(c.mag() - 1.) > eps) {
    std::cout << " Rotation not good by bector mag: " << (a).mag() << ", " << (b).mag() << ", " << (c).mag()
              << " for det at pos " << det.surface().position() << std::endl;
  }
}

void TrackerDigiGeometryAnalyzer::checkTopology(const GeomDetUnit& det) {
  const StripTopology& topol = dynamic_cast<const StripTopology&>(det.topology());
  const int N = 5;

  std::cout << std::endl << "Topology test along strip 0" << std::endl;
  LocalVector stripDir = topol.localPosition(MeasurementPoint(0, 0.5)) - topol.localPosition(MeasurementPoint(0, 0));
  std::cout << "StripDir " << stripDir << std::endl;
  for (int i = -N; i <= N; i++) {
    MeasurementPoint mp(0, float(i) / float(2 * N));
    LocalPoint lp = topol.localPosition(mp);
    double strlen = topol.localStripLength(lp);
    LocalError le = topol.localError(mp, MeasurementError(0.25, 0, 0.25));
    LocalError lep = le.rotate(stripDir.y(), stripDir.x());
    LocalError lem = le.rotate(stripDir.y(), -stripDir.x());
    GlobalPoint gp(det.surface().toGlobal(lp));
    std::cout << "gpos (r,phi) (" << gp.perp() << ", " << gp.phi() << ") lpos " << lp << " lerr.x (0,+,-) "
              << sqrt(le.xx()) << "," << sqrt(lep.xx()) << "," << sqrt(lem.xx()) << " strlen " << strlen << std::endl;
  }

  std::cout << std::endl << "Topology test along middle strip" << std::endl;
  float midstrip = topol.strip(LocalPoint(0, 0));
  for (int i = -N; i <= N; i++) {
    MeasurementPoint mp(midstrip, float(i) / float(2 * N));
    LocalPoint lp = topol.localPosition(mp);
    double strlen = topol.localStripLength(lp);
    LocalError le = topol.localError(mp, MeasurementError(0.25, 0, 0.25));
    GlobalPoint gp(det.surface().toGlobal(lp));
    std::cout << "gpos (r,phi) (" << gp.perp() << ", " << gp.phi() << ") lpos " << lp << " lerr.x " << sqrt(le.xx())
              << " strlen " << strlen << std::endl;
  }
}

std::ostream& TrackerDigiGeometryAnalyzer::cylindrical(std::ostream& os, const GlobalPoint& gp) const {
  os << "(" << gp.perp() << "," << gp.phi() << "," << gp.z();
  return os;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerDigiGeometryAnalyzer);
