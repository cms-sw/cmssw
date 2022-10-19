
#include <memory>
#include <string>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

class L1MuonOverlapParamsDBProducer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  L1MuonOverlapParamsDBProducer(const edm::ParameterSet&);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> esTokenParams_;
  std::unique_ptr<L1TMuonOverlapParams> omtfParams;
};

L1MuonOverlapParamsDBProducer::L1MuonOverlapParamsDBProducer(const edm::ParameterSet&)
    : esTokenParams_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "params"))) {}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void L1MuonOverlapParamsDBProducer::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {
  omtfParams = std::make_unique<L1TMuonOverlapParams>(iSetup.getData(esTokenParams_));
  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }
}
void L1MuonOverlapParamsDBProducer::endRun(const edm::Run&, const edm::EventSetup&) {}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void L1MuonOverlapParamsDBProducer::analyze(const edm::Event&, const edm::EventSetup&) {
  std::string recordName = "L1TMuonOverlapParamsRcd";
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(*omtfParams, poolDbService->currentTime(), recordName);
  }
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MuonOverlapParamsDBProducer);
