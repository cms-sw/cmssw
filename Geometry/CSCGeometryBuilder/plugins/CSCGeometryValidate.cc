/*
//\class CSCGeometryValidate

 Description: CSC GeometryValidate from DD & DD4hep

//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Thu, 05 March 2020 
*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"
#include "Geometry/CSCGeometry/interface/CSCWireTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Fireworks/Core/interface/FWGeometry.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <TFile.h>
#include <TH1.h>

#include <limits>
#include <string>
#include <type_traits>
#include <algorithm>

using namespace std;

template <class T>
typename enable_if<!numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return abs(x - y) <= numeric_limits<T>::epsilon() * abs(x + y) * ulp
         // unless the result is subnormal
         || abs(x - y) < numeric_limits<T>::min();
}

using namespace edm;

class CSCGeometryValidate : public one::EDAnalyzer<> {
public:
  explicit CSCGeometryValidate(const ParameterSet&);
  ~CSCGeometryValidate() override {}

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void validateCSCChamberGeometry();
  void validateCSCLayerGeometry();

  void compareTransform(const GlobalPoint&, const TGeoMatrix*);
  void compareShape(const GeomDet*, const float*);

  float getDistance(const GlobalPoint&, const GlobalPoint&);
  float getDiff(const float, const float);

  void makeHistograms(const char*);
  void makeHistograms2(const char*);
  void makeHistogram(const string&, vector<float>&);

  void clearData() {
    globalDistances_.clear();
    topWidths_.clear();
    bottomWidths_.clear();
    lengths_.clear();
    thicknesses_.clear();
  }

  void clearData2() {
    yAxisOrientation_.clear();
    sOffset_.clear();
    yCentreOfStripPlane_.clear();
    angularWidth_.clear();
    centreToIntersection_.clear();
    phiOfOneEdge_.clear();
    wireSpacing_.clear();
    wireAngle_.clear();
  }

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> tokCSC_;
  const CSCGeometry* cscGeometry_;
  FWGeometry fwGeometry_;
  TFile* outFile_;
  //chambers
  vector<float> globalDistances_;
  vector<float> topWidths_;
  vector<float> bottomWidths_;
  vector<float> lengths_;
  vector<float> thicknesses_;
  // strips
  vector<float> yAxisOrientation_;
  vector<float> sOffset_;
  vector<float> yCentreOfStripPlane_;
  vector<float> angularWidth_;
  vector<float> centreToIntersection_;
  vector<float> phiOfOneEdge_;
  //wires
  vector<float> wireSpacing_;
  vector<float> wireAngle_;
  //files
  string infileName_;
  string outfileName_;
  int tolerance_;
};

CSCGeometryValidate::CSCGeometryValidate(const edm::ParameterSet& iConfig)
    : tokCSC_{esConsumes<CSCGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      infileName_(iConfig.getUntrackedParameter<string>("infileName", "cmsGeom10.root")),
      outfileName_(iConfig.getUntrackedParameter<string>("outfileName", "validateCSCGeometry.root")),
      tolerance_(iConfig.getUntrackedParameter<int>("tolerance", 6)) {
  fwGeometry_.loadMap(infileName_.c_str());
  outFile_ = TFile::Open(outfileName_.c_str(), "RECREATE");
}

void CSCGeometryValidate::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  cscGeometry_ = &eventSetup.getData(tokCSC_);
  LogVerbatim("CSCGeometry") << "Validating CSC chamber geometry";
  validateCSCChamberGeometry();
  validateCSCLayerGeometry();
}

void CSCGeometryValidate::validateCSCChamberGeometry() {
  clearData();

  for (auto const& it : cscGeometry_->chambers()) {
    CSCDetId chId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(chId.rawId());

    if (!matrix) {
      LogVerbatim("CSCGeometry") << "Failed to get matrix of CSC chamber with detid: " << chId.rawId();
      continue;
    }
    compareTransform(gp, matrix);

    auto const& shape = fwGeometry_.getShapePars(chId.rawId());

    if (!shape) {
      LogVerbatim("CSCGeometry") << "Failed to get shape of CSC chamber with detid: " << chId.rawId();
      continue;
    }
    compareShape(it, shape);
  }
  makeHistograms("CSC Chamber");
}

void CSCGeometryValidate::validateCSCLayerGeometry() {
  clearData2();

  for (auto const& it : cscGeometry_->layers()) {
    CSCDetId chId = it->id();
    const CSCLayerGeometry* laygeo = it->geometry();
    const int n_strips = laygeo->numberOfStrips();
    const int n_wire = laygeo->numberOfWires();
    const float strips_offset = laygeo->stripOffset();
    const CSCStripTopology* stopo = laygeo->topology();
    const float ycentre_of_strip_plane = stopo->yCentreOfStripPlane();
    const float angular_width = stopo->angularWidth();
    const float y_axis_orientation = stopo->yAxisOrientation();
    const float centre_to_intersection = stopo->centreToIntersection();
    const float phi_of_one_edge = stopo->phiOfOneEdge();
    const float* parameters = fwGeometry_.getParameters(chId.rawId());
    const CSCWireTopology* wiretopo = laygeo->wireTopology();
    const double wire_spacing = wiretopo->wireSpacing();
    const float wire_angle = wiretopo->wireAngle();

    if (n_strips) {
      for (int istrips = 1; istrips <= n_strips; istrips++) {
        yAxisOrientation_.push_back(fabs(y_axis_orientation - parameters[0]));
        sOffset_.push_back(fabs(strips_offset - parameters[4]));
        yCentreOfStripPlane_.push_back(fabs(ycentre_of_strip_plane - parameters[2]));
        angularWidth_.push_back(fabs(angular_width - parameters[5]));
        centreToIntersection_.push_back(fabs(centre_to_intersection - parameters[1]));
        phiOfOneEdge_.push_back(fabs(phi_of_one_edge - parameters[3]));
      }
    } else {
      LogVerbatim("CSCGeometry") << "ATTENTION! nStrips == 0";
    }

    if (n_wire) {
      for (int iwires = 1; iwires <= n_wire; iwires++) {
        wireSpacing_.push_back(fabs(wire_spacing - parameters[6]));
        wireAngle_.push_back(fabs(wire_angle - parameters[7]));
      }
    } else {
      LogVerbatim("CSCGeometry") << "ATTENTION! nWires == 0";
    }
  }
  makeHistograms2("CSC Layer");
}

void CSCGeometryValidate::compareTransform(const GlobalPoint& gp, const TGeoMatrix* matrix) {
  double local[3] = {0.0, 0.0, 0.0};
  double global[3];

  matrix->LocalToMaster(local, global);

  float distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  if ((distance >= 0.0) && (distance < 1.0e-7))
    distance = 0.0;  // set a tollerance for the distance inside Histos
  globalDistances_.push_back(distance);
}

void CSCGeometryValidate::compareShape(const GeomDet* det, const float* shape) {
  float shapeTopWidth;
  float shapeBottomWidth;
  float shapeLength;
  float shapeThickness;

  if (shape[0] == 1) {
    shapeTopWidth = shape[2];
    shapeBottomWidth = shape[1];
    shapeLength = shape[4];
    shapeThickness = shape[3];
  } else if (shape[0] == 2) {
    shapeTopWidth = shape[1];
    shapeBottomWidth = shape[1];
    shapeLength = shape[2];
    shapeThickness = shape[3];
  } else {
    LogVerbatim("CSCGeometry") << "Failed to get box or trapezoid from shape";

    return;
  }

  float topWidth, bottomWidth;
  float length, thickness;

  const Bounds* bounds = &(det->surface().bounds());
  if (const TrapezoidalPlaneBounds* tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds)) {
    array<const float, 4> const& ps = tpbs->parameters();

    assert(ps.size() == 4);

    bottomWidth = ps[0];
    topWidth = ps[1];
    thickness = ps[2];
    length = ps[3];
  } else if ((dynamic_cast<const RectangularPlaneBounds*>(bounds))) {
    length = det->surface().bounds().length() * 0.5;
    topWidth = det->surface().bounds().width() * 0.5;
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness() * 0.5;
  } else {
    LogVerbatim("CSCGeometry") << "Failed to get bounds";

    return;
  }
  topWidths_.push_back(fabs(shapeTopWidth - topWidth));
  bottomWidths_.push_back(fabs(shapeBottomWidth - bottomWidth));
  lengths_.push_back(fabs(shapeLength - length));
  thicknesses_.push_back(fabs(shapeThickness - thickness));
}

float CSCGeometryValidate::getDistance(const GlobalPoint& p1, const GlobalPoint& p2) {
  return sqrt((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
              (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

float CSCGeometryValidate::getDiff(const float val1, const float val2) {
  if (almost_equal(val1, val2, tolerance_))
    return 0.0f;
  else
    return (val1 - val2);
}

void CSCGeometryValidate::makeHistograms(const char* detector) {
  outFile_->cd();

  string d(detector);

  string gdn = d + ": distance between points in global coordinates";
  makeHistogram(gdn, globalDistances_);

  string twn = d + ": absolute difference between top widths (along X)";
  makeHistogram(twn, topWidths_);

  string bwn = d + ": absolute difference between bottom widths (along X)";
  makeHistogram(bwn, bottomWidths_);

  string ln = d + ": absolute difference between lengths (along Y)";
  makeHistogram(ln, lengths_);

  string tn = d + ": absolute difference between thicknesses (along Z)";
  makeHistogram(tn, thicknesses_);
}

void CSCGeometryValidate::makeHistograms2(const char* detector) {
  outFile_->cd();

  string d(detector);

  string ns = d + ": absolute difference between Y Axis Orientation of the Strips";
  makeHistogram(ns, yAxisOrientation_);

  string pi = d + ": absolute difference between Strips Offset";
  makeHistogram(pi, sOffset_);

  string pl = d + ": absolute difference between 'Y centre' of the Strips Planes";
  makeHistogram(pl, yCentreOfStripPlane_);

  string aw = d + ": absolute difference between 'angular width' of the Strips ";
  makeHistogram(aw, angularWidth_);

  string ci = d + ": absolute difference between 'centre to intersection' of the Strips ";
  makeHistogram(ci, centreToIntersection_);

  string po = d + ": absolute difference between 'phi of one edge' of the Strips ";
  makeHistogram(po, phiOfOneEdge_);

  string ws = d + ": absolute difference between 'wire spacing' of the Wires ";
  makeHistogram(ws, wireSpacing_);

  string wa = d + ": absolute difference between 'wire angle' of the Wires ";
  makeHistogram(wa, wireAngle_);
}

void CSCGeometryValidate::makeHistogram(const string& name, vector<float>& data) {
  if (data.empty())
    return;

  const auto [minE, maxE] = minmax_element(begin(data), end(data));

  TH1D hist(name.c_str(), name.c_str(), 100, *minE * (1 + 0.10), *maxE * (1 + 0.10));

  for (auto const& it : data)
    hist.Fill(it);

  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void CSCGeometryValidate::beginJob() { outFile_->cd(); }

void CSCGeometryValidate::endJob() {
  LogVerbatim("CSCGeometry") << "Done.";
  LogVerbatim("CSCGeometry") << "Results written to " << outfileName_;
  outFile_->Close();
}

DEFINE_FWK_MODULE(CSCGeometryValidate);
