/*
//\class GEMGeometryValidate

 Description: GEM GeometryValidate from DD & DD4HEP
 
 //
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  27 Jan 2020 
*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
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
#include <cmath>

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

class GEMGeometryValidate : public one::EDAnalyzer<> {
public:
  explicit GEMGeometryValidate(const ParameterSet&);
  ~GEMGeometryValidate() override {}

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void validateGEMChamberGeometry();
  void validateGEMEtaPartitionGeometry();

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
    nstrips_.clear();
    pitch_.clear();
    stripslen_.clear();
  }

  edm::ESHandle<GEMGeometry> gemGeometry_;
  FWGeometry fwGeometry_;
  TFile* outFile_;
  vector<float> globalDistances_;
  vector<float> topWidths_;
  vector<float> bottomWidths_;
  vector<float> lengths_;
  vector<float> thicknesses_;
  vector<float> nstrips_;
  vector<float> pitch_;
  vector<float> stripslen_;
  string infileName_;
  string outfileName_;
  int tolerance_;
};

GEMGeometryValidate::GEMGeometryValidate(const edm::ParameterSet& iConfig)
    : infileName_(iConfig.getUntrackedParameter<string>("infileName", "cmsRecoGeom-2021.root")),
      outfileName_(iConfig.getUntrackedParameter<string>("outfileName", "validateGEMGeometry.root")),
      tolerance_(iConfig.getUntrackedParameter<int>("tolerance", 6)) {
  fwGeometry_.loadMap(infileName_.c_str());
  outFile_ = TFile::Open(outfileName_.c_str(), "RECREATE");
}

void GEMGeometryValidate::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  eventSetup.get<MuonGeometryRecord>().get(gemGeometry_);

  if (gemGeometry_.isValid()) {
    LogVerbatim("GEMGeometry") << "Validating GEM chamber geometry";

    validateGEMChamberGeometry();

    validateGEMEtaPartitionGeometry();

  } else
    LogVerbatim("GEMGeometry") << "Invalid GEM geometry";
}

void GEMGeometryValidate::validateGEMChamberGeometry() {
  clearData();

  for (auto const& it : gemGeometry_->chambers()) {
    GEMDetId chId = it->id();
    GlobalPoint gp = it->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(chId.rawId());

    if (!matrix) {
      LogVerbatim("GEMGeometry") << "Failed to get matrix of GEM chamber with detid: " << chId.rawId();
      continue;
    }
    compareTransform(gp, matrix);

    auto const& shape = fwGeometry_.getShapePars(chId.rawId());

    if (!shape) {
      LogVerbatim("GEMGeometry") << "Failed to get shape of GEM chamber with detid: " << chId.rawId();
      continue;
    }
    compareShape(it, shape);
  }
  makeHistograms("GEM Chamber");
}

void GEMGeometryValidate::validateGEMEtaPartitionGeometry() {
  clearData2();

  for (auto const& it : gemGeometry_->etaPartitions()) {
    GEMDetId chId = it->id();
    const int n_strips = it->nstrips();
    const float n_pitch = it->pitch();
    const StripTopology& topo = it->specificTopology();
    const float stripLen = topo.stripLength();
    const float* parameters = fwGeometry_.getParameters(chId.rawId());
    nstrips_.push_back(abs(n_strips - parameters[0]));

    if (n_strips) {
      for (int istrips = 1; istrips <= n_strips; istrips++) {
        pitch_.push_back(fabs(n_pitch - parameters[2]));

        stripslen_.push_back(fabs(stripLen - parameters[1]));
      }
    } else {
      LogVerbatim("GEMGeometry") << "ATTENTION! nStrips == 0";
    }
  }
  makeHistograms2("GEM Eta Partition");
}

void GEMGeometryValidate::compareTransform(const GlobalPoint& gp, const TGeoMatrix* matrix) {
  double local[3] = {0.0, 0.0, 0.0};
  double global[3];

  matrix->LocalToMaster(local, global);

  float distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  if (abs(distance) < 1.0e-7)
    distance = 0.0;  // set a tollerance for the distance inside Histos
  globalDistances_.push_back(distance);
}

void GEMGeometryValidate::compareShape(const GeomDet* det, const float* shape) {
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
    LogVerbatim("GEMGeometry") << "Failed to get box or trapezoid from shape";

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
    LogVerbatim("GEMGeometry") << "Failed to get bounds";

    return;
  }
  topWidths_.push_back(fabs(shapeTopWidth - topWidth));
  bottomWidths_.push_back(fabs(shapeBottomWidth - bottomWidth));
  lengths_.push_back(fabs(shapeLength - length));
  thicknesses_.push_back(fabs(shapeThickness - thickness));
}

float GEMGeometryValidate::getDistance(const GlobalPoint& p1, const GlobalPoint& p2) {
  return sqrt((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
              (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

float GEMGeometryValidate::getDiff(const float val1, const float val2) {
  if (almost_equal(val1, val2, tolerance_))
    return 0.0f;
  else
    return (val1 - val2);
}

void GEMGeometryValidate::makeHistograms(const char* detector) {
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

void GEMGeometryValidate::makeHistograms2(const char* detector) {
  outFile_->cd();

  string d(detector);

  string ns = d + ": absolute difference between nStrips";
  makeHistogram(ns, nstrips_);

  string pi = d + ": absolute difference between Strips Pitch";
  makeHistogram(pi, pitch_);

  string pl = d + ": absolute difference between Strips Length";
  makeHistogram(pl, stripslen_);
}

void GEMGeometryValidate::makeHistogram(const string& name, vector<float>& data) {
  if (data.empty())
    return;

  const auto [minE, maxE] = minmax_element(begin(data), end(data));

  TH1D hist(name.c_str(), name.c_str(), 100, *minE * (1 + 0.10), *maxE * (1 + 0.10));

  for (auto const& it : data)
    hist.Fill(it);

  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void GEMGeometryValidate::beginJob() { outFile_->cd(); }

void GEMGeometryValidate::endJob() {
  LogVerbatim("GEMGeometry") << "Done.";
  LogVerbatim("GEMGeometry") << "Results written to " << outfileName_;
  outFile_->Close();
}

DEFINE_FWK_MODULE(GEMGeometryValidate);
