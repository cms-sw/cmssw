#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "HeterogeneousCore/CUDATest/interface/Thing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"

#include "TestCUDAAnalyzerGPUKernel.h"

class TestCUDAAnalyzerGPU : public edm::global::EDAnalyzer<> {
public:
  explicit TestCUDAAnalyzerGPU(edm::ParameterSet const& iConfig);
  ~TestCUDAAnalyzerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;
  void endJob() override;

private:
  std::string const label_;
  edm::EDGetTokenT<cms::cuda::Product<cms::cudatest::Thing>> const srcToken_;
  double const minValue_;
  double const maxValue_;
  // the public interface is thread safe
  CMS_THREAD_SAFE mutable std::unique_ptr<TestCUDAAnalyzerGPUKernel> gpuAlgo_;
};

TestCUDAAnalyzerGPU::TestCUDAAnalyzerGPU(edm::ParameterSet const& iConfig)
    : label_(iConfig.getParameter<std::string>("@module_label")),
      srcToken_(consumes<cms::cuda::Product<cms::cudatest::Thing>>(iConfig.getParameter<edm::InputTag>("src"))),
      minValue_(iConfig.getParameter<double>("minValue")),
      maxValue_(iConfig.getParameter<double>("maxValue")) {
  edm::Service<CUDAInterface> cuda;
  if (cuda and cuda->enabled()) {
    auto streamPtr = cms::cuda::getStreamCache().get();
    gpuAlgo_ = std::make_unique<TestCUDAAnalyzerGPUKernel>(streamPtr.get());
  }
}

void TestCUDAAnalyzerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of cms::cuda::Product<cms::cudatest::Thing>.");
  desc.add<double>("minValue", -1e308);
  desc.add<double>("maxValue", 1e308);
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDAnalyzer is part of the TestCUDAProducer* family. It models a GPU analyzer.");
}

void TestCUDAAnalyzerGPU::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::analyze begin event "
                                          << iEvent.id().event() << " stream " << iEvent.streamID();

  auto const& in = iEvent.get(srcToken_);
  cms::cuda::ScopedContextAnalyze ctx{in};
  cms::cudatest::Thing const& input = ctx.get(in);
  gpuAlgo_->analyzeAsync(input.get(), ctx.stream());

  edm::LogVerbatim("TestCUDAAnalyzerGPU")
      << label_ << " TestCUDAAnalyzerGPU::analyze end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAAnalyzerGPU::endJob() {
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::endJob begin";

  auto streamPtr = cms::cuda::getStreamCache().get();
  auto value = gpuAlgo_->value(streamPtr.get());
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << "  accumulated value " << value;
  assert(minValue_ <= value && value <= maxValue_);

  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::endJob end";
}

DEFINE_FWK_MODULE(TestCUDAAnalyzerGPU);
