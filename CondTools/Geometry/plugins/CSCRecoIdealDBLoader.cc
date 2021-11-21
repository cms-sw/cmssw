#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/CSCGeometryBuilder/interface/CSCGeometryParsFromDD.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class CSCRecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  CSCRecoIdealDBLoader(edm::ParameterSet const&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> muonGeomConstantsToken_;
};

CSCRecoIdealDBLoader::CSCRecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4hep_ = iC.getUntrackedParameter<bool>("fromDD4hep", false);
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  muonGeomConstantsToken_ = esConsumes<edm::Transition::BeginRun>();
}

void CSCRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("CSCRecoIdealDBLoader") << "CSCRecoIdealDBLoader::beginRun";

  RecoIdealGeometry rig;
  CSCRecoDigiParameters rdp;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("CSCRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  auto pMNDC = es.getHandle(muonGeomConstantsToken_);
  CSCGeometryParsFromDD cscgp;

  if (fromDD4hep_) {
    auto pDD = es.getTransientHandle(dd4HepCompactViewToken_);
    const cms::DDCompactView& cpv = *pDD;
    cscgp.build(&cpv, *pMNDC, rig, rdp);
  } else {
    auto pDD = es.getTransientHandle(compactViewToken_);
    const DDCompactView& cpv = *pDD;
    cscgp.build(&cpv, *pMNDC, rig, rdp);
  }

  if (mydbservice->isNewTagRequest("CSCRecoGeometryRcd")) {
    mydbservice->createOneIOV(rig, mydbservice->beginOfTime(), "CSCRecoGeometryRcd");
  } else {
    edm::LogError("CSCRecoIdealDBLoader") << "CSCRecoGeometryRcd Tag is already present.";
  }
  if (mydbservice->isNewTagRequest("CSCRecoDigiParametersRcd")) {
    mydbservice->createOneIOV(rdp, mydbservice->beginOfTime(), "CSCRecoDigiParametersRcd");
  } else {
    edm::LogError("CSCRecoIdealDBLoader") << "CSCRecoDigiParametersRcd Tag is already present.";
  }
}

DEFINE_FWK_MODULE(CSCRecoIdealDBLoader);
