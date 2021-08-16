#include <L1Trigger/L1TMuonOverlapPhase1/plugins/L1MuonOverlapPhase1ParamsDBProducer.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

L1MuonOverlapPhase1ParamsDBProducer::L1MuonOverlapPhase1ParamsDBProducer(const edm::ParameterSet& cfg)
    : omtfParamsEsToken(esConsumes<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd, edm::Transition::BeginRun>()) {
  edm::LogImportant("L1MuonOverlapParamsDBProducer") << " L1MuonOverlapPhase1ParamsDBProducer() " << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void L1MuonOverlapPhase1ParamsDBProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  /*
  const L1TMuonOverlapParamsRcd& omtfParamsRcd = iSetup.get<L1TMuonOverlapParamsRcd>();

  edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;

  omtfParamsRcd.get(omtfParamsHandle);

  omtfParams = std::unique_ptr<L1TMuonOverlapParams>(new L1TMuonOverlapParams(*omtfParamsHandle.product()));*/

  omtfParams = std::unique_ptr<L1TMuonOverlapParams>(new L1TMuonOverlapParams(iSetup.getData(omtfParamsEsToken)));

  if (!omtfParams) {
    edm::LogError("L1MuonOverlapParamsDBProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  edm::LogImportant("L1MuonOverlapParamsDBProducer") << " beginRun() " << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void L1MuonOverlapPhase1ParamsDBProducer::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::LogImportant("L1MuonOverlapParamsDBProducer") << " analyze() line " << __LINE__ << std::endl;
  std::string recordName = "L1TMuonOverlapParamsRcd";
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(omtfParams.get(), poolDbService->currentTime(), recordName);
  }
  edm::LogImportant("L1MuonOverlapParamsDBProducer") << " analyze() line " << __LINE__ << std::endl;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MuonOverlapPhase1ParamsDBProducer);
