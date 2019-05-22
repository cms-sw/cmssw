#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryParsFromDD.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

class ME0RecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  ME0RecoIdealDBLoader(const edm::ParameterSet&) {}

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
};

void ME0RecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("ME0RecoIdealDBLoader") << "ME0RecoIdealDBLoader::beginRun";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("ME0RecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  if (mydbservice->isNewTagRequest("ME0RecoGeometryRcd")) {
    edm::ESTransientHandle<DDCompactView> pDD;
    edm::ESHandle<MuonDDDConstants> pMNDC;
    es.get<IdealGeometryRecord>().get(pDD);
    es.get<MuonNumberingRecord>().get(pMNDC);

    const DDCompactView& cpv = *pDD;
    ME0GeometryParsFromDD me0pd;

    RecoIdealGeometry* rig = new RecoIdealGeometry;
    me0pd.build(&cpv, *pMNDC, *rig);

    mydbservice->createNewIOV<RecoIdealGeometry>(
        rig, mydbservice->beginOfTime(), mydbservice->endOfTime(), "ME0RecoGeometryRcd");
  } else {
    edm::LogError("ME0RecoIdealDBLoader") << "ME0RecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE(ME0RecoIdealDBLoader);
