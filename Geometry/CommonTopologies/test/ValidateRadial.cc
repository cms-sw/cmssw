#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyStripTopology.h"
#include "Geometry/CommonTopologies/interface/TkRadialStripTopology.h"
#include "TFile.h"
#include "TProfile.h"

class ValidateRadial : public edm::one::EDAnalyzer<> {
public:
  ValidateRadial(const edm::ParameterSet&);
  ~ValidateRadial() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::vector<const TkRadialStripTopology*> get_list_of_radial_topologies(const edm::Event&, const edm::EventSetup&);
  void test_topology(const TkRadialStripTopology*, unsigned);
  bool pass_frame_change_test(const TkRadialStripTopology* t, float strip, float stripErr2, bool);
  bool EQUAL(const double a, const double b) { return fabs(a - b) < epsilon_; }
  const double epsilon_;
  TFile* file_;
  const bool printOut_;
  const bool posOnly_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokTrackerGeometry_;

  mutable float maxerrU = 0.;
  mutable float maxerrUV = 0.;
};

ValidateRadial::ValidateRadial(const edm::ParameterSet& cfg)
    : epsilon_(cfg.getParameter<double>("Epsilon")),
      file_(new TFile(cfg.getParameter<std::string>("FileName").c_str(), "RECREATE")),
      printOut_(cfg.getParameter<bool>("PrintOut")),
      posOnly_(cfg.getParameter<bool>("PosOnly")),
      tokTrackerGeometry_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {
  std::cout << "I'm ALIVE" << std::endl;
}

void ValidateRadial::analyze(const edm::Event& e, const edm::EventSetup& es) {
  std::vector<const TkRadialStripTopology*> topologies = get_list_of_radial_topologies(e, es);
  for (unsigned i = 0; i < topologies.size(); i++) {
    test_topology(topologies[i], i);
  }
  file_->Close();
}

std::vector<const TkRadialStripTopology*> ValidateRadial::get_list_of_radial_topologies(const edm::Event& e,
                                                                                        const edm::EventSetup& es) {
  std::vector<const TkRadialStripTopology*> topos;
  auto theTrackerGeometry = es.getHandle(tokTrackerGeometry_);
  const uint32_t radial_detids[] = {402666125,   //TID r1
                                    402668833,   //TID r2
                                    402673476,   //TID r3
                                    470066725,   //TEC r1
                                    470390853,   //TEC r2
                                    470114664,   //TEC r3
                                    470131344,   //TEC r4
                                    470079661,   //TEC r5
                                    470049476,   //TEC r6
                                    470045428};  //TEC r7
  for (unsigned int radial_detid : radial_detids) {
    auto g = dynamic_cast<const StripGeomDetUnit*>(theTrackerGeometry->idToDet(radial_detid));
    if (!g)
      std::cout << "no geom for " << radial_detid << std::endl;
    auto const topol = &g->specificTopology();
    const TkRadialStripTopology* rt = nullptr;
    auto const proxyT = dynamic_cast<const ProxyStripTopology*>(topol);
    if (proxyT)
      rt = dynamic_cast<const TkRadialStripTopology*>(&(proxyT->specificTopology()));
    else
      rt = dynamic_cast<const TkRadialStripTopology*>(topol);
    if (!rt)
      std::cout << "no radial topology for " << radial_detid << std::endl;
    else
      topos.emplace_back(rt);
  }
  return topos;
}

#include "Geometry/CommonTopologies/interface/CSCRadialStripTopology.h"

void compare(const TkRadialStripTopology& t,
             const CSCRadialStripTopology& ot,
             const float strip,
             const float stripErr2) {
  const LocalPoint lp = t.localPosition(strip);
  const LocalError le = t.localError(strip, stripErr2);

  const MeasurementPoint mp = t.measurementPosition(lp);
  const MeasurementError me = t.measurementError(lp, le);
  const float newstrip = t.strip(lp);

  const LocalPoint olp = ot.localPosition(strip);
  const LocalError ole = ot.localError(strip, stripErr2);

  const MeasurementPoint omp = ot.measurementPosition(lp);
  const MeasurementError ome = ot.measurementError(lp, le);
  const float onewstrip = ot.strip(lp);

  if (fabs(lp.x() - olp.x()) > 0.001)
    std::cout << "FAILED " << lp.x() << " " << olp.x() << std::endl;
  if (fabs(lp.y() - olp.y()) > 0.001)
    std::cout << "FAILED " << lp.y() << " " << olp.y() << std::endl;
  if (fabs(newstrip - onewstrip) > 0.001)
    std::cout << "FAILED " << newstrip << " " << onewstrip << std::endl;
  if (fabs(le.xx() - ole.xx()) > 0.001)
    std::cout << "FAILED " << le.xx() << " " << ole.xx() << std::endl;
  if (fabs(le.yy() - ole.yy()) > 0.001)
    std::cout << "FAILED " << le.yy() << " " << ole.xy() << std::endl;
  if (fabs(le.xy() - ole.xy()) > 0.001)
    std::cout << "FAILED " << le.xy() << " " << ole.yy() << std::endl;
  if (fabs(mp.x() - omp.x()) > 0.001)
    std::cout << "FAILED " << mp.x() << " " << omp.x() << std::endl;
  if (fabs(me.uu() - ome.uu()) > 0.001)
    std::cout << "FAILED " << me.uu() << " " << ome.uu() << std::endl;
}

void ValidateRadial::test_topology(const TkRadialStripTopology* t, unsigned i) {
  CSCRadialStripTopology oldt(t->nstrips(),
                              t->angularWidth(),
                              t->detHeight(),
                              t->centreToIntersection(),
                              t->yAxisOrientation(),
                              t->yCentreOfStripPlane());

  std::cout << "\nTK\n" << *t << std::endl;
  std::cout << "\nCSC\n" << oldt << std::endl;

  TProfile prof(("se2limit1" + std::to_string(i)).c_str(),
                "Precision Limit of recoverable strip error (1st order);strip;strip error",
                t->nstrips() / 8,
                0,
                t->nstrips());
  TProfile prof2(("se2limit2" + std::to_string(i)).c_str(),
                 "Precision Limit of recoverable strip error (2nd order);strip;strip error",
                 t->nstrips() / 8,
                 0,
                 t->nstrips());
  for (float strip = 0; strip < t->nstrips(); strip += 0.5) {
    for (float stripErr2 = 0.03; stripErr2 > 1e-10; stripErr2 /= 1.1) {
      compare(*t, oldt, strip, stripErr2);
      if (!pass_frame_change_test(t, strip, stripErr2, false)) {
        prof.Fill(strip, sqrt(stripErr2));
        break;
      }
    }
    for (float stripErr2 = 0.03; stripErr2 > 1e-10; stripErr2 /= 1.1)
      if (!pass_frame_change_test(t, strip, stripErr2, true)) {
        prof2.Fill(strip, sqrt(stripErr2));
        break;
      }
  }
  prof.Write();
  prof2.Write();
}

ValidateRadial::~ValidateRadial() {
  std::cout << "ValidateRadial max UU, max UV " << maxerrU << " " << maxerrUV << std::endl;
}

bool ValidateRadial::pass_frame_change_test(const TkRadialStripTopology* t,
                                            const float strip,
                                            const float stripErr2,
                                            const bool secondOrder) {
  const LocalPoint lp = t->localPosition(strip);
  const LocalError le = t->localError(strip, stripErr2);
  const MeasurementPoint mp = t->measurementPosition(lp);
  const MeasurementError me = t->measurementError(lp, le);
  const float newstrip = t->strip(lp);
  const LocalPoint newlp = t->localPosition(mp);
  const LocalError newle = t->localError(mp, me);
  const MeasurementPoint newmp = t->measurementPosition(newlp);
  const MeasurementError newme = t->measurementError(newlp, newle);

  maxerrU = std::max(maxerrU, std::abs(me.uu() - stripErr2) / stripErr2);
  maxerrUV = std::max(maxerrUV, std::abs(me.uv()));

  const bool pass1p = fabs(strip - newstrip) < 0.001 && fabs(strip - mp.x()) < 0.001 && fabs(mp.x() - newstrip) < 0.001;
  const bool pass1e =
      (fabs(me.uu() - stripErr2) / stripErr2 < epsilon_) && (fabs(me.uv()) < epsilon_) && me.uu() > 0 && me.vv() > 0;
  const bool pass2p = fabs(strip - newmp.x()) < 0.001;
  const bool pass2e = (fabs(newme.uu() - stripErr2) / stripErr2 < epsilon_) && (fabs(newme.uv()) < epsilon_) &&
                      newme.uu() > 0 && newme.vv() > 0;

  const bool passp = (secondOrder ? pass2p : pass1p);
  const bool passe = (secondOrder ? pass2e : pass1e);

  if (printOut_ && ((!passp) || ((!posOnly_) && (!passe))))
    std::cout << "FAILED "
              << "(" << strip << ", " << newstrip << ", " << mp.x() << ")\t"
              << "(" << stripErr2 << ", " << me.uu() << ")\t\t" << (me.uv()) << std::endl;
  return passp & passe;
}

DEFINE_FWK_MODULE(ValidateRadial);
