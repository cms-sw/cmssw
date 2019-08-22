#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryParsFromDD.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

class DTRecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  DTRecoIdealDBLoader(edm::ParameterSet const&) {}

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
};

void DTRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  RecoIdealGeometry* rig = new RecoIdealGeometry;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("DTRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(pDD);
  es.get<MuonNumberingRecord>().get(pMNDC);

  const DDCompactView& cpv = *pDD;
  DTGeometryParsFromDD dtgp;

  dtgp.build(&cpv, *pMNDC, *rig);

  if (mydbservice->isNewTagRequest("DTRecoGeometryRcd")) {
    mydbservice->createNewIOV<RecoIdealGeometry>(
        rig, mydbservice->beginOfTime(), mydbservice->endOfTime(), "DTRecoGeometryRcd");
  } else {
    edm::LogError("DTRecoIdealDBLoader") << "DTRecoGeometryRcd Tag is already present.";
  }
}

DEFINE_FWK_MODULE(DTRecoIdealDBLoader);
