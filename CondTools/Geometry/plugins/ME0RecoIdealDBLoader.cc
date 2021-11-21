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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/GEMGeometryBuilder/interface/ME0GeometryParsFromDD.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class ME0RecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  ME0RecoIdealDBLoader(const edm::ParameterSet&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> muonGeomConstantsToken_;
};

ME0RecoIdealDBLoader::ME0RecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4hep_ = iC.getUntrackedParameter<bool>("fromDD4hep", false);  // set true for DD4hep
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  muonGeomConstantsToken_ = esConsumes<edm::Transition::BeginRun>();
}

void ME0RecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("ME0RecoIdealDBLoader") << "ME0RecoIdealDBLoader::beginRun";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("ME0RecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  if (mydbservice->isNewTagRequest("ME0RecoGeometryRcd")) {
    auto pMNDC = es.getHandle(muonGeomConstantsToken_);
    ME0GeometryParsFromDD me0pd;
    RecoIdealGeometry rig;
    if (fromDD4hep_) {
      edm::LogVerbatim("ME0RecoIdealDBLoader") << "(0) ME0RecoIdealDBLoader - DD4hep ";
      auto pDD = es.getTransientHandle(dd4HepCompactViewToken_);
      const cms::DDCompactView& cpv = *pDD;
      me0pd.build(&cpv, *pMNDC, rig);
    } else {
      edm::LogVerbatim("ME0RecoIdealDBLoader") << "(0) ME0RecoIdealDBLoader - DDD ";
      auto pDD = es.getTransientHandle(compactViewToken_);
      const DDCompactView& cpv = *pDD;
      me0pd.build(&cpv, *pMNDC, rig);
    }
    mydbservice->createOneIOV(rig, mydbservice->beginOfTime(), "ME0RecoGeometryRcd");
  } else {
    edm::LogError("ME0RecoIdealDBLoader") << "ME0RecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE(ME0RecoIdealDBLoader);
