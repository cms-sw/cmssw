#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/GEMGeometryBuilder/interface/GEMGeometryParsFromDD.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMRecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  GEMRecoIdealDBLoader(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> muonGeomConstantsToken_;
};

GEMRecoIdealDBLoader::GEMRecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4hep_ = iC.getUntrackedParameter<bool>("fromDD4hep", false);  // set true for DD4hep
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  muonGeomConstantsToken_ = esConsumes<edm::Transition::BeginRun>();
}

void GEMRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("GEMRecoIdealDBLoader") << "GEMRecoIdealDBLoader::beginRun";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("GEMRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  if (mydbservice->isNewTagRequest("GEMRecoGeometryRcd")) {
    auto pMNDC = es.getHandle(muonGeomConstantsToken_);

    GEMGeometryParsFromDD rpcpd;
    RecoIdealGeometry rig;

    if (fromDD4hep_) {
      edm::LogVerbatim("GEMRecoIdealDBLoader") << "(0) GEMRecoIdealDBLoader - DD4hep ";
      auto pDD = es.getTransientHandle(dd4HepCompactViewToken_);
      const cms::DDCompactView& cpv = *pDD;
      rpcpd.build(&cpv, *pMNDC, rig);
    } else {
      edm::LogVerbatim("GEMRecoIdealDBLoader") << "(0) GEMRecoIdealDBLoader - DDD ";
      auto pDD = es.getTransientHandle(compactViewToken_);
      const DDCompactView& cpv = *pDD;
      rpcpd.build(&cpv, *pMNDC, rig);
    }
    mydbservice->createOneIOV(rig, mydbservice->beginOfTime(), "GEMRecoGeometryRcd");
  } else {
    edm::LogError("GEMRecoIdealDBLoader") << "GEMRecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE(GEMRecoIdealDBLoader);
