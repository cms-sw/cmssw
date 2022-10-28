#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"

class PTrackerParametersDBBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  PTrackerParametersDBBuilder(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
};

PTrackerParametersDBBuilder::PTrackerParametersDBBuilder(const edm::ParameterSet& iConfig) {
  fromDD4hep_ = iConfig.getParameter<bool>("fromDD4hep");
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
}

void PTrackerParametersDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  PTrackerParameters ptp;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("PTrackerParametersDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  TrackerParametersFromDD builder;

  if (!fromDD4hep_) {
    auto cpv = es.getTransientHandle(compactViewToken_);
    builder.build(&(*cpv), ptp);
  } else {
    auto cpv = es.getTransientHandle(dd4HepCompactViewToken_);
    builder.build(&(*cpv), ptp);
  }

  if (mydbservice->isNewTagRequest("PTrackerParametersRcd")) {
    mydbservice->createOneIOV(ptp, mydbservice->beginOfTime(), "PTrackerParametersRcd");
  } else {
    edm::LogError("PTrackerParametersDBBuilder") << "PTrackerParameters and PTrackerParametersRcd Tag already present";
  }
}

DEFINE_FWK_MODULE(PTrackerParametersDBBuilder);
