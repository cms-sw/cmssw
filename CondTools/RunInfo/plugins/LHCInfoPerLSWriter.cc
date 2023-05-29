#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"

#include <memory>
#include <iostream>

class LHCInfoPerLSWriter : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoPerLSWriter(const edm::ParameterSet&) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}
};

void LHCInfoPerLSWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LHCInfoPerLS lhcInfoPerLS;
  lhcInfoPerLS.setFillNumber(7066);
  lhcInfoPerLS.setLumiSection(1);
  lhcInfoPerLS.setCrossingAngleX(170);
  lhcInfoPerLS.setCrossingAngleY(170);
  lhcInfoPerLS.setBetaStarX(11);
  lhcInfoPerLS.setBetaStarY(11);
  lhcInfoPerLS.setRunNumber(301765);

  // store the info into a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(lhcInfoPerLS, poolDbService->currentTime(), "LHCInfoPerLSRcd");
  else
    throw cms::Exception("LHCInfoPerLSWriter") << "PoolDBService required.";
}

DEFINE_FWK_MODULE(LHCInfoPerLSWriter);
