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
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryParsFromDD.h"
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
  bool fromDD4Hep_;
};

CSCRecoIdealDBLoader::CSCRecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4Hep_ = iC.getParameter<bool>("fromDD4Hep");
}

void CSCRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("CSCRecoIdealDBLoader") << "CSCRecoIdealDBLoader::beginRun";

  RecoIdealGeometry* rig = new RecoIdealGeometry;
  CSCRecoDigiParameters* rdp = new CSCRecoDigiParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("CSCRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  edm::ESHandle<MuonGeometryConstants> pMNDC;
  CSCGeometryParsFromDD cscgp;

  if (fromDD4Hep_) {
    edm::ESTransientHandle<cms::DDCompactView> pDD;
    es.get<IdealGeometryRecord>().get(pDD);
    es.get<IdealGeometryRecord>().get(pMNDC);
    const cms::DDCompactView& cpv = *pDD;
    cscgp.build(&cpv, *pMNDC, *rig, *rdp);
  } else {
    edm::ESTransientHandle<DDCompactView> pDD;
    es.get<IdealGeometryRecord>().get(pDD);
    es.get<IdealGeometryRecord>().get(pMNDC);
    const DDCompactView& cpv = *pDD;
    cscgp.build(&cpv, *pMNDC, *rig, *rdp);
  }

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
