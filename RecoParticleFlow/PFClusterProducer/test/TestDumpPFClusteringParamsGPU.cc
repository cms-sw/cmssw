#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

#include "testKernels.h"

class TestDumpPFClusteringParamsGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestDumpPFClusteringParamsGPU(edm::ParameterSet const&);
  ~TestDumpPFClusteringParamsGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

  edm::ESGetToken<PFClusteringParamsGPU, JobConfigurationGPURecord> const pfClusParamsToken_;

  cms::cuda::ContextState cudaState_;
};

TestDumpPFClusteringParamsGPU::TestDumpPFClusteringParamsGPU(edm::ParameterSet const& iConfig)
    : pfClusParamsToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("pfClusteringParameters"))} {}

void TestDumpPFClusteringParamsGPU::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription psetDesc;
  psetDesc.add<edm::ESInputTag>("pfClusteringParameters",
                                edm::ESInputTag("pfClusteringParamsGPUESSource", "pfClusParamsOffline"));
  desc.addWithDefaultLabel(psetDesc);
}

void TestDumpPFClusteringParamsGPU::acquire(edm::Event const& event,
                                            edm::EventSetup const& setup,
                                            edm::WaitingTaskWithArenaHolder holder) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};
  auto const& pfClusParamsProduct = setup.getData(pfClusParamsToken_).getProduct(ctx.stream());
  testPFlow::testPFClusteringParamsEntryPoint(pfClusParamsProduct, ctx.stream());
}

void TestDumpPFClusteringParamsGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestDumpPFClusteringParamsGPU);
