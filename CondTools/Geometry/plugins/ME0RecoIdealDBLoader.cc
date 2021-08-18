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
  bool fromDD4Hep_;
};

ME0RecoIdealDBLoader::ME0RecoIdealDBLoader(const edm::ParameterSet& iC) {
  fromDD4Hep_ = iC.getUntrackedParameter<bool>("fromDD4Hep", false);  // set true for DD4HEP
}

void ME0RecoIdealDBLoader::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("ME0RecoIdealDBLoader") << "ME0RecoIdealDBLoader::beginRun";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("ME0RecoIdealDBLoader") << "PoolDBOutputService unavailable";
    return;
  }

  if (mydbservice->isNewTagRequest("ME0RecoGeometryRcd")) {
    edm::ESHandle<MuonGeometryConstants> pMNDC;
    ME0GeometryParsFromDD me0pd;
    RecoIdealGeometry* rig = new RecoIdealGeometry;
    if (fromDD4Hep_) {
      edm::LogVerbatim("ME0RecoIdealDBLoader") << "(0) ME0RecoIdealDBLoader - DD4HEP ";
      edm::ESTransientHandle<cms::DDCompactView> pDD;
      es.get<IdealGeometryRecord>().get(pDD);
      es.get<IdealGeometryRecord>().get(pMNDC);
      const cms::DDCompactView& cpv = *pDD;
      me0pd.build(&cpv, *pMNDC, *rig);
    } else {
      edm::LogVerbatim("ME0RecoIdealDBLoader") << "(0) ME0RecoIdealDBLoader - DDD ";
      edm::ESTransientHandle<DDCompactView> pDD;
      es.get<IdealGeometryRecord>().get(pDD);
      es.get<IdealGeometryRecord>().get(pMNDC);
      const DDCompactView& cpv = *pDD;
      me0pd.build(&cpv, *pMNDC, *rig);
    }
    mydbservice->createNewIOV<RecoIdealGeometry>(
        rig, mydbservice->beginOfTime(), mydbservice->endOfTime(), "ME0RecoGeometryRcd");
  } else {
    edm::LogError("ME0RecoIdealDBLoader") << "ME0RecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE(ME0RecoIdealDBLoader);
