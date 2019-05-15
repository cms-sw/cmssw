#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "TestGPUConcurrency.h"
#include "TestGPUConcurrencyAlgo.h"

TestGPUConcurrency::TestGPUConcurrency(edm::ParameterSet const& config):
  HeterogeneousEDProducer(config),
  blocks_(config.getParameter<uint32_t>("blocks")),
  threads_(config.getParameter<uint32_t>("threads")),
  sleep_(config.getParameter<uint32_t>("sleep"))
{
}

void TestGPUConcurrency::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  HeterogeneousEDProducer::fillPSetDescription(desc);
  desc.add<uint32_t>("blocks", 100);
  desc.add<uint32_t>("threads", 256);
  desc.add<uint32_t>("sleep", 1000000);
  descriptions.add("testGPUConcurrency", desc);
}

void TestGPUConcurrency::beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream)
{
  algo_ = new TestGPUConcurrencyAlgo(blocks_, threads_, sleep_);
}

void TestGPUConcurrency::acquireGPUCuda(const edm::HeterogeneousEvent& event, const edm::EventSetup& setup, cuda::stream_t<>& cudaStream)
{
  algo_->kernelWrapper(cudaStream.id());
}

void TestGPUConcurrency::produceCPU(edm::HeterogeneousEvent& event, const edm::EventSetup& setup)
{
}

void TestGPUConcurrency::produceGPUCuda(edm::HeterogeneousEvent& event, const edm::EventSetup& setup, cuda::stream_t<>& cudaStream)
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestGPUConcurrency);
