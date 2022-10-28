#include "L1MuonOverlapPhase1ParamsDBProducer.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <memory>

L1MuonOverlapPhase1ParamsDBProducer::L1MuonOverlapPhase1ParamsDBProducer(const edm::ParameterSet& cfg)
    : omtfParamsEsToken(esConsumes<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd, edm::Transition::BeginRun>()) {
  edm::LogVerbatim("L1MuonOverlapParamsDBProducer") << " L1MuonOverlapPhase1ParamsDBProducer() " << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void L1MuonOverlapPhase1ParamsDBProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  omtfParams = std::make_unique<L1TMuonOverlapParams>(iSetup.getData(omtfParamsEsToken));

  if (!omtfParams) {
    edm::LogError("L1MuonOverlapParamsDBProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  edm::LogVerbatim("L1MuonOverlapParamsDBProducer") << " beginRun() " << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void L1MuonOverlapPhase1ParamsDBProducer::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::LogVerbatim("L1MuonOverlapParamsDBProducer") << " analyze() line " << __LINE__ << std::endl;
  std::string recordName = "L1TMuonOverlapParamsRcd";
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(*omtfParams, poolDbService->currentTime(), recordName);
  }
  edm::LogVerbatim("L1MuonOverlapParamsDBProducer") << " analyze() line " << __LINE__ << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MuonOverlapPhase1ParamsDBProducer);
