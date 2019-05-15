#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "tbb/concurrent_vector.h"

#include <chrono>
#include <future>
#include <random>
#include <thread>

/**
 * The purpose of this test is to demonstrate the following
 * - EDProducer implementing an algorithm for CPU and a (mock GPU) device
 *   * The mock device exercises all the structures without a need for actual device
 * - How to read heterogeneous product from event
 * - How to read normal product from event
 * - How to write heterogeneous product to event
 */
class TestHeterogeneousEDProducerGPUMock: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
                                                                           heterogeneous::GPUMock,
                                                                           heterogeneous::CPU
                                                                           > > {
public:
  explicit TestHeterogeneousEDProducerGPUMock(edm::ParameterSet const& iConfig);
  ~TestHeterogeneousEDProducerGPUMock() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUMockProduct<unsigned int> >;

  std::string label_;
  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;
  edm::EDGetTokenT<int> srcIntToken_;

  // hack for GPU mock
  tbb::concurrent_vector<std::future<void> > pendingFutures;

  // simulating GPU memory
  unsigned int gpuOutput_;

  void acquireGPUMock(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, std::function<void()> callback) override;

  void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;
  void produceGPUMock(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;
};

TestHeterogeneousEDProducerGPUMock::TestHeterogeneousEDProducerGPUMock(edm::ParameterSet const& iConfig):
  HeterogeneousEDProducer(iConfig),
  label_(iConfig.getParameter<std::string>("@module_label"))
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumesHeterogeneous(srcTag);
  }
  auto srcIntTag = iConfig.getParameter<edm::InputTag>("srcInt");
  if(!srcIntTag.label().empty()) {
    srcIntToken_ = consumes<int>(srcIntTag);
  }

  produces<HeterogeneousProduct>();
  produces<int>();
  produces<int>("foo");
}

void TestHeterogeneousEDProducerGPUMock::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  desc.add<edm::InputTag>("srcInt", edm::InputTag());
  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.add("testHeterogeneousEDProducerGPUMock", desc);
}

void TestHeterogeneousEDProducerGPUMock::acquireGPUMock(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, std::function<void()> callback) {
  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << " " << label_ << " TestHeterogeneousEDProducerGPUMock::acquireGPUMock begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  unsigned int input = 0;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<unsigned int> hin;
    iEvent.getByToken<OutputType>(srcToken_, hin);
    input = *hin;
  }
  if(!srcIntToken_.isUninitialized()) {
    edm::Handle<int> hin;
    iEvent.getByToken(srcIntToken_, hin);
    input += *hin;
  }

  /// GPU work
  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(0.1, 1.0);
  auto dur = dist(gen);
  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << "  " << label_ << " Task (GPU) for event " << iEvent.id().event() << " in stream " << iEvent.streamID() << " will take " << dur << " seconds";

  auto ret = std::async(std::launch::async,
                        [this, dur, input,
                         callback = std::move(callback),
                         eventId = iEvent.id().event(),
                         streamId = iEvent.streamID()
                         ](){
                          std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
                          gpuOutput_ = input + streamId*100 + eventId;
                          edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << "   " << label_ << " TestHeterogeneousEDProducerGPUMock::acquireGPUMock finished async for event " << eventId << " stream " << streamId;
                          callback();
                        });
  pendingFutures.push_back(std::move(ret));

  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << " " << label_ << " TestHeterogeneousEDProducerGPUMock::acquireGPUMock end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestHeterogeneousEDProducerGPUMock::produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << label_ << " TestHeterogeneousEDProducerGPUMock::produceCPU begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  unsigned int input = 0;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<unsigned int> hin;
    iEvent.getByToken<OutputType>(srcToken_, hin);
    input = *hin;
  }
  if(!srcIntToken_.isUninitialized()) {
    edm::Handle<int> hin;
    iEvent.getByToken(srcIntToken_, hin);
    input += *hin;
  }

  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(1.0, 3.0);
  auto dur = dist(gen);
  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << "  Task (CPU) for event " << iEvent.id().event() << " in stream " << iEvent.streamID() << " will take " << dur << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

  const unsigned int output = input+ iEvent.streamID()*100 + iEvent.id().event();

  iEvent.put<OutputType>(std::make_unique<unsigned int>(output));
  iEvent.put(std::make_unique<int>(1));
  iEvent.put(std::make_unique<int>(2), "foo");

  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << label_ << " TestHeterogeneousEDProducerGPUMock::produceCPU end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << output;
}

void TestHeterogeneousEDProducerGPUMock::produceGPUMock(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << label_ << " TestHeterogeneousEDProducerGPUMock::produceGPUMock begin event " << iEvent.id().event() << " stream " << iEvent.streamID();

  iEvent.put<OutputType>(std::make_unique<unsigned int>(gpuOutput_),
                         [this,
                          eventId = iEvent.id().event(),
                          streamId = iEvent.streamID()
                          ](const unsigned int& src, unsigned int& dst) {
                           edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << "  " << label_ << " Task (GPU) for event " << eventId << " in stream " << streamId << " copying to CPU";
                           dst = src;
                         });
  iEvent.put(std::make_unique<int>(2));

  edm::LogPrint("TestHeterogeneousEDProducerGPUMock") << label_ << " TestHeterogeneousEDProducerGPUMock::produceGPUMock end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << gpuOutput_;
}

DEFINE_FWK_MODULE(TestHeterogeneousEDProducerGPUMock);
