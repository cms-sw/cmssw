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
  bool fromDD4Hep_;
};

GEMRecoIdealDBLoader::GEMRecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4Hep_ = iC.getUntrackedParameter<bool>("fromDD4Hep", false);  // set true for DD4HEP
}

void GEMRecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("GEMRecoIdealDBLoader") << "GEMRecoIdealDBLoader::beginRun";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("GEMRecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  if (mydbservice->isNewTagRequest("GEMRecoGeometryRcd")) {
    edm::ESHandle<MuonGeometryConstants> pMNDC;
    GEMGeometryParsFromDD rpcpd;
    RecoIdealGeometry* rig = new RecoIdealGeometry;

    if (fromDD4Hep_) {
      edm::LogVerbatim("GEMRecoIdealDBLoader") << "(0) GEMRecoIdealDBLoader - DD4HEP ";
      edm::ESTransientHandle<cms::DDCompactView> pDD;
      es.get<IdealGeometryRecord>().get(pDD);
      es.get<IdealGeometryRecord>().get(pMNDC);
      const cms::DDCompactView& cpv = *pDD;
      rpcpd.build(&cpv, *pMNDC, *rig);
    } else {
      edm::LogVerbatim("GEMRecoIdealDBLoader") << "(0) GEMRecoIdealDBLoader - DDD ";
      edm::ESTransientHandle<DDCompactView> pDD;
      es.get<IdealGeometryRecord>().get(pDD);
      es.get<IdealGeometryRecord>().get(pMNDC);
      const DDCompactView& cpv = *pDD;
      rpcpd.build(&cpv, *pMNDC, *rig);
    }
    mydbservice->createNewIOV<RecoIdealGeometry>(
        rig, mydbservice->beginOfTime(), mydbservice->endOfTime(), "GEMRecoGeometryRcd");
  } else {
    edm::LogError("GEMRecoIdealDBLoader") << "GEMRecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE(GEMRecoIdealDBLoader);
