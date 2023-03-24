#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/RecoGeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/DTGeometryBuilderinterface/DTGeometryParsFromDD.h"
#include "Geometry/Records/interface/RecoIdealGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

class DTRecoIdealDBLoader : public edm::one::EDAnalyzer<> {
public:
  explicit DTRecoIdealDBLoader(const edm::ParameterSet& iConfig);
  ~DTRecoIdealDBLoader();

  void beginJob() override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override;

private:
  const std::string label_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> tokDT_;
  const edm::ESGetToken<DDCompactView, IdealGeometryRecord> tokDDD_;
  int rotNumSeed_;
};

using namespace std;

DTRecoIdealDBLoader::DTRecoIdealDBLoader(const edm::ParameterSet& iConfig)
    : label_(),
      tokDT_{esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      tokDDD_{esConsumes<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{"", label_})} {
  edm::LogVerbatim("DTGeometry") << "DTRecoIdealDBLoader::DTRecoIdealDBLoader";
}

DTRecoIdealDBLoader::~DTRecoIdealDBLoader() { edm::LogVerbatim("DTGeometry") << "DTRecoIdealDBLoader::~DTRecoIdealDBLoader"; }

void DTRecoIdealDBLoader::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  edm::LogVerbatim("DTGeometry") << "DTRecoIdealDBLoader::beginJob";
  RecoIdealGeometry rig;

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogVerbatim("DTGeometry") << "PoolDBOutputService unavailable";
    return;
  }

  const DDCompactView& cpv = es.getData(tokDDD_);
  const auto& pMNDC = &es.getData(tokDT_);
  DTGeometryParsFromDD dtgp;

  dtgp.build(&cpv, *pMNDC, rig);
  edm::LogVerbatim("DTGeometry") << "RecoIdealGeometry " << rig.size();

  if (mydbservice->isNewTagRequest("RecoIdealGeometryRcd")) {
    edm::LogVerbatim("DTGeometry") << "mydbservice " << mydbservice->beginOfTime() << " to " << mydbservice->endOfTime();
    mydbservice->createOneIOV(rig, mydbservice->beginOfTime(), "RecoIdealGeometryRcd");
  } else {
    edm::LogVerbatim("DTGeometry") << "RecoIdealGeometryRcd Tag is already present.";
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTRecoIdealDBLoader);
