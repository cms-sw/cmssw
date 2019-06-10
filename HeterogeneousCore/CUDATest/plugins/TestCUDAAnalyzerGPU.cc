#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"

#include "TestCUDAAnalyzerGPUKernel.h"

class TestCUDAAnalyzerGPU: public edm::global::EDAnalyzer<> {
public:
  explicit TestCUDAAnalyzerGPU(const edm::ParameterSet& iConfig);
  ~TestCUDAAnalyzerGPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  void endJob() override;

private:
  std::string label_;
  edm::EDGetTokenT<CUDAProduct<CUDAThing>> srcToken_;
  double minValue_;
  double maxValue_;
  TestCUDAAnalyzerGPUKernel gpuAlgo_;
};

TestCUDAAnalyzerGPU::TestCUDAAnalyzerGPU(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcToken_(consumes<CUDAProduct<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src"))),
  minValue_(iConfig.getParameter<double>("minValue")),
  maxValue_(iConfig.getParameter<double>("maxValue"))
{}

void TestCUDAAnalyzerGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Source of CUDAProduct<CUDAThing>.");
  desc.add<double>("minValue", -1e308);
  desc.add<double>("maxValue", 1e308);
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDAnalyzer is part of the TestCUDAProducer* family. It models a GPU analyzer.");
}

void TestCUDAAnalyzerGPU::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::analyze begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  CUDAScopedContext ctx{iEvent.streamID()}; // creating a new CUDA stream
  const CUDAThing& input = ctx.get(iEvent.get(srcToken_));
  gpuAlgo_.analyzeAsync(input.get(), ctx.stream());

  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::analyze end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAAnalyzerGPU::endJob() {
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::endJob begin";

  auto value = gpuAlgo_.value();
  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << "  accumulated value " << value;
  assert(minValue_ <= value && value <= maxValue_);

  edm::LogVerbatim("TestCUDAAnalyzerGPU") << label_ << " TestCUDAAnalyzerGPU::endJob end";
}

DEFINE_FWK_MODULE(TestCUDAAnalyzerGPU);
