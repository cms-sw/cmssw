#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using RotationMatrix = ROOT::Math::Rotation3D;

class PGeometricDetBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  PGeometricDetBuilder(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  void putOne(const GeometricDet* gd, PGeometricDet* pgd, int lev);
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geometricDetToken_;
};

PGeometricDetBuilder::PGeometricDetBuilder(const edm::ParameterSet& iConfig) {
  fromDD4hep_ = iConfig.getParameter<bool>("fromDD4hep");
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  geometricDetToken_ = esConsumes<edm::Transition::BeginRun>();
}

void PGeometricDetBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  PGeometricDet pgd;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("PGeometricDetBuilder") << "PoolDBOutputService unavailable";
    return;
  }
  if (!fromDD4hep_) {
    auto pDD = es.getTransientHandle(compactViewToken_);
  } else {
    auto pDD = es.getTransientHandle(dd4HepCompactViewToken_);
  }
  const GeometricDet* tracker = &es.getData(geometricDetToken_);

  // so now I have the tracker itself. loop over all its components to store them.
  putOne(tracker, &pgd, 0);
  std::vector<const GeometricDet*> tc = tracker->components();
  std::vector<const GeometricDet*>::const_iterator git = tc.begin();
  std::vector<const GeometricDet*>::const_iterator egit = tc.end();
  int lev = 1;
  for (; git != egit; ++git) {  // one level below "tracker"
    putOne(*git, &pgd, lev);
    std::vector<const GeometricDet*> inone = (*git)->components();
    std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
    std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
    ++lev;
    for (; git2 != egit2; ++git2) {  // level 2
      putOne(*git2, &pgd, lev);
      std::vector<const GeometricDet*> intwo = (*git2)->components();
      std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
      std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
      ++lev;
      for (; git3 != egit3; ++git3) {  // level 3
        putOne(*git3, &pgd, lev);
        std::vector<const GeometricDet*> inthree = (*git3)->components();
        std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
        std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
        ++lev;
        for (; git4 != egit4; ++git4) {  //level 4
          putOne(*git4, &pgd, lev);
          std::vector<const GeometricDet*> infour = (*git4)->components();
          std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
          std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
          ++lev;
          for (; git5 != egit5; ++git5) {  // level 5
            putOne(*git5, &pgd, lev);
            std::vector<const GeometricDet*> infive = (*git5)->components();
            std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
            std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
            ++lev;
            for (; git6 != egit6; ++git6) {  //level 6
              putOne(*git6, &pgd, lev);
              std::vector<const GeometricDet*> insix = (*git6)->components();
            }  // level 6
            --lev;
          }  // level 5
          --lev;
        }  // level 4
        --lev;
      }  //level 3
      --lev;
    }  // level 2
    --lev;
  }
  if (mydbservice->isNewTagRequest("IdealGeometryRecord")) {
    mydbservice->createOneIOV(pgd, mydbservice->beginOfTime(), "IdealGeometryRecord");
  } else {
    edm::LogError("PGeometricDetBuilder") << "PGeometricDetBuilder Tag already present";
  }
}

void PGeometricDetBuilder::putOne(const GeometricDet* gd, PGeometricDet* pgd, int lev) {
  PGeometricDet::Item item;
  const Translation& tran = gd->translation();
  const RotationMatrix& rot = gd->rotation();
  DD3Vector x, y, z;
  rot.GetComponents(x, y, z);
  item._name = gd->name();
  item._ns = std::string();
  item._level = lev;
  using cms_rounding::roundIfNear0;
  const double tol = 1.e-10;
  // Round very small calculated values to 0 to avoid discrepancies
  // between +0 and -0 in comparisons.
  item._x = roundIfNear0(tran.X(), tol);
  item._y = roundIfNear0(tran.Y(), tol);
  item._z = roundIfNear0(tran.Z(), tol);
  item._phi = gd->phi();
  item._rho = gd->rho();
  item._a11 = roundIfNear0(x.X(), tol);
  item._a12 = roundIfNear0(y.X(), tol);
  item._a13 = roundIfNear0(z.X(), tol);
  item._a21 = roundIfNear0(x.Y(), tol);
  item._a22 = roundIfNear0(y.Y(), tol);
  item._a23 = roundIfNear0(z.Y(), tol);
  item._a31 = roundIfNear0(x.Z(), tol);
  item._a32 = roundIfNear0(y.Z(), tol);
  item._a33 = roundIfNear0(z.Z(), tol);
  item._shape = static_cast<int>(gd->shape_dd4hep());
  item._type = gd->type();
  if (gd->shape_dd4hep() == cms::DDSolidShape::ddbox) {
    item._params0 = gd->params()[0];
    item._params1 = gd->params()[1];
    item._params2 = gd->params()[2];
    item._params3 = 0;
    item._params4 = 0;
    item._params5 = 0;
    item._params6 = 0;
    item._params7 = 0;
    item._params8 = 0;
    item._params9 = 0;
    item._params10 = 0;
  } else if (gd->shape_dd4hep() == cms::DDSolidShape::ddtrap) {
    item._params0 = gd->params()[0];
    item._params1 = gd->params()[1];
    item._params2 = gd->params()[2];
    item._params3 = gd->params()[3];
    item._params4 = gd->params()[4];
    item._params5 = gd->params()[5];
    item._params6 = gd->params()[6];
    item._params7 = gd->params()[7];
    item._params8 = gd->params()[8];
    item._params9 = gd->params()[9];
    item._params10 = gd->params()[10];
  } else {
    item._params0 = 0;
    item._params1 = 0;
    item._params2 = 0;
    item._params3 = 0;
    item._params4 = 0;
    item._params5 = 0;
    item._params6 = 0;
    item._params7 = 0;
    item._params8 = 0;
    item._params9 = 0;
    item._params10 = 0;
  }
  item._geographicalID = gd->geographicalId();
  item._radLength = gd->radLength();
  item._xi = gd->xi();
  item._pixROCRows = gd->pixROCRows();
  item._pixROCCols = gd->pixROCCols();
  item._pixROCx = gd->pixROCx();
  item._pixROCy = gd->pixROCy();
  item._stereo = gd->stereo();
  item._siliconAPVNum = gd->siliconAPVNum();

  GeometricDet::nav_type const& nt = gd->navType();
  size_t nts = nt.size();
  item._numnt = nts;
  std::vector<int> tempnt(nt.begin(), nt.end());
  for (size_t extrant = nt.size(); extrant < 11; ++extrant) {
    tempnt.push_back(-1);
  }
  item._nt0 = tempnt[0];
  item._nt1 = tempnt[1];
  item._nt2 = tempnt[2];
  item._nt3 = tempnt[3];
  item._nt4 = tempnt[4];
  item._nt5 = tempnt[5];
  item._nt6 = tempnt[6];
  item._nt7 = tempnt[7];
  item._nt8 = tempnt[8];
  item._nt9 = tempnt[9];
  item._nt10 = tempnt[10];

  pgd->pgeomdets_.push_back(item);
}

DEFINE_FWK_MODULE(PGeometricDetBuilder);
