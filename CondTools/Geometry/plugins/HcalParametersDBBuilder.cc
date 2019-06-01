#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"

class HcalParametersDBBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  HcalParametersDBBuilder(const edm::ParameterSet&) {}

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
};

void HcalParametersDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  HcalParameters* php = new HcalParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("HcalParametersDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }
  edm::ESTransientHandle<DDCompactView> cpv;
  es.get<IdealGeometryRecord>().get(cpv);

  HcalParametersFromDD builder;
  builder.build(&(*cpv), *php);

  if (mydbservice->isNewTagRequest("HcalParametersRcd")) {
    mydbservice->createNewIOV<HcalParameters>(
        php, mydbservice->beginOfTime(), mydbservice->endOfTime(), "HcalParametersRcd");
  } else {
    edm::LogError("HcalParametersDBBuilder") << "HcalParameters and HcalParametersRcd Tag already present";
  }
}

DEFINE_FWK_MODULE(HcalParametersDBBuilder);
