#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <chrono>
#include <random>
#include <thread>

class TestCUDAProducerCPU : public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerCPU(edm::ParameterSet const& iConfig);
  ~TestCUDAProducerCPU() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID id, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  std::string const label_;
  edm::EDGetTokenT<int> srcToken_;
  edm::EDPutTokenT<int> const dstToken_;
};

TestCUDAProducerCPU::TestCUDAProducerCPU(edm::ParameterSet const& iConfig)
    : label_{iConfig.getParameter<std::string>("@module_label")}, dstToken_{produces<int>()} {
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if (!srcTag.label().empty()) {
    srcToken_ = consumes<int>(srcTag);
  }
}

void TestCUDAProducerCPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag())->setComment("Optional source of another TestCUDAProducerCPU.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer is part of the TestCUDAProducer* family. It models a CPU algorithm.");
}

void TestCUDAProducerCPU::produce(edm::StreamID id, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  edm::LogVerbatim("TestCUDAProducerCPU")
      << label_ << " TestCUDAProducerCPU::produce begin event " << iEvent.id().event() << " stream " << id;

  int input = 0;
  if (!srcToken_.isUninitialized()) {
    input = iEvent.get(srcToken_);
  }

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(0.2, 1.5);
  auto dur = dist(gen);
  edm::LogVerbatim("TestCUDAProducerCPU")
      << " Task (CPU) for event " << iEvent.id().event() << " in stream " << id << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1) * dur);

  unsigned int const output = input + id * 100 + iEvent.id().event();

  iEvent.emplace(dstToken_, output);

  edm::LogVerbatim("TestCUDAProducerCPU") << label_ << " TestCUDAProducerCPU::produce end event " << iEvent.id().event()
                                          << " stream " << id << " result " << output;
}

DEFINE_FWK_MODULE(TestCUDAProducerCPU);
