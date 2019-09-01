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
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryParsFromDD.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

class CSCRecoIdealDBLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  CSCRecoIdealDBLoader(edm::ParameterSet const&) {}

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
};

void CSCRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("CSCRecoIdealDBLoader") << "CSCRecoIdealDBLoader::beginRun";

  RecoIdealGeometry* rig = new RecoIdealGeometry;
  CSCRecoDigiParameters* rdp = new CSCRecoDigiParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("CSCRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(pDD);
  es.get<MuonNumberingRecord>().get(pMNDC);

  const DDCompactView& cpv = *pDD;
  CSCGeometryParsFromDD cscgp;

  cscgp.build(&cpv, *pMNDC, *rig, *rdp);

  if (mydbservice->isNewTagRequest("CSCRecoGeometryRcd")) {
    mydbservice->createNewIOV<RecoIdealGeometry>(
        rig, mydbservice->beginOfTime(), mydbservice->endOfTime(), "CSCRecoGeometryRcd");
  } else {
    edm::LogError("CSCRecoIdealDBLoader") << "CSCRecoGeometryRcd Tag is already present.";
  }
  if (mydbservice->isNewTagRequest("CSCRecoDigiParametersRcd")) {
    mydbservice->createNewIOV<CSCRecoDigiParameters>(
        rdp, mydbservice->beginOfTime(), mydbservice->endOfTime(), "CSCRecoDigiParametersRcd");
  } else {
    edm::LogError("CSCRecoIdealDBLoader") << "CSCRecoDigiParametersRcd Tag is already present.";
  }
}

DEFINE_FWK_MODULE(CSCRecoIdealDBLoader);
