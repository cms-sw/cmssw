#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"

//#define EDM_ML_DEBUG

class HcalParametersDBBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  HcalParametersDBBuilder(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4Hep_;
};

HcalParametersDBBuilder::HcalParametersDBBuilder(const edm::ParameterSet& ps)
    : fromDD4Hep_(ps.getParameter<bool>("fromDD4Hep")) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalParametersDBBuilder::HcalParametersDBBuilder called with dd4hep: "
                               << fromDD4Hep_;
#endif
}

void HcalParametersDBBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("HcalParametersWriter", desc);
}

void HcalParametersDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  HcalParameters* php = new HcalParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("HcalParametersDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  HcalParametersFromDD builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "HcalParametersDBBuilder::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    builder.build((*cpv), *php);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "HcalParametersDBBuilder::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    builder.build(&(*cpv), *php);
  }

  if (mydbservice->isNewTagRequest("HcalParametersRcd")) {
    mydbservice->createNewIOV<HcalParameters>(
        php, mydbservice->beginOfTime(), mydbservice->endOfTime(), "HcalParametersRcd");
  } else {
    edm::LogError("HcalParametersDBBuilder") << "HcalParameters and HcalParametersRcd Tag already present";
  }
}

DEFINE_FWK_MODULE(HcalParametersDBBuilder);
