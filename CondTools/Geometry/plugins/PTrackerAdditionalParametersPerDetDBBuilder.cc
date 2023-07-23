#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class PTrackerAdditionalParametersPerDetDBBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  PTrackerAdditionalParametersPerDetDBBuilder(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
};

PTrackerAdditionalParametersPerDetDBBuilder::PTrackerAdditionalParametersPerDetDBBuilder(
    const edm::ParameterSet& iConfig)
    : geomDetToken_(esConsumes<edm::Transition::BeginRun>()) {}

void PTrackerAdditionalParametersPerDetDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  PTrackerAdditionalParametersPerDet ptitp;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("PTrackerAdditionalParametersPerDetDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  const GeometricDet* gd = &es.getData(geomDetToken_);

  std::vector<const GeometricDet*> comp;
  gd->deepComponents(comp);

  for (auto& i : comp) {
    ptitp.setGeographicalId(i->geographicalId());
  }

  if (mydbservice->isNewTagRequest("PTrackerAdditionalParametersPerDetRcd")) {
    mydbservice->createOneIOV(ptitp, mydbservice->beginOfTime(), "PTrackerAdditionalParametersPerDetRcd");
  } else {
    edm::LogError("PTrackerAdditionalParametersPerDetDBBuilder")
        << "PTrackerAdditionalParametersPerDet and PTrackerAdditionalParametersPerDetRcd Tag already present";
  }
}

DEFINE_FWK_MODULE(PTrackerAdditionalParametersPerDetDBBuilder);
