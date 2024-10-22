#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/RPCGeometryBuilder/interface/RPCGeometryParsFromDD.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class RPCRecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  RPCRecoIdealDBLoader(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> muonGeomConstantsToken_;
};

RPCRecoIdealDBLoader::RPCRecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4hep_ = iC.getUntrackedParameter<bool>("fromDD4hep", false);
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  muonGeomConstantsToken_ = esConsumes<edm::Transition::BeginRun>();
}

void RPCRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  RecoIdealGeometry rig;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("RPCRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  auto pMNDC = es.getHandle(muonGeomConstantsToken_);
  RPCGeometryParsFromDD rpcpd;

  if (fromDD4hep_) {
    auto pDD = es.getTransientHandle(dd4HepCompactViewToken_);
    const cms::DDCompactView& cpv = *pDD;
    rpcpd.build(&cpv, *pMNDC, rig);
  } else {
    auto pDD = es.getTransientHandle(compactViewToken_);
    const DDCompactView& cpv = *pDD;
    rpcpd.build(&cpv, *pMNDC, rig);
  }
  if (mydbservice->isNewTagRequest("RPCRecoGeometryRcd")) {
    mydbservice->createOneIOV(rig, mydbservice->beginOfTime(), "RPCRecoGeometryRcd");
  } else {
    edm::LogError("RPCRecoIdealDBLoader") << "RPCRecoGeometryRcd Tag is already present.";
  }
}

DEFINE_FWK_MODULE(RPCRecoIdealDBLoader);
