#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "tbb/concurrent_vector.h"

#include <chrono>
#include <future>
#include <random>
#include <thread>

namespace {
  // hack for GPU mock
  tbb::concurrent_vector<std::future<void> > pendingFutures;

  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUMockProduct<unsigned int> >;

  class TestAlgo {
  public:
    TestAlgo() {}
    ~TestAlgo() = default;

    void setInput(const OutputType *input, unsigned int eventId, unsigned int streamId) {
      input_ = input;
      eventId_ = eventId;
      streamId_ = streamId;
    }
    
    void runCPU() {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
      auto dur = dist(gen);
      edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "   Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(1)*dur);

      auto input = input_ ? input_->getProduct<HeterogeneousDevice::kCPU>() : 0U;

      output_ = input + streamId_*100 + eventId_;
    }

    void runGPUMock(std::function<void()> callback) {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(0.1, 1.0); 
      auto dur = dist(gen);
      edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";
      ranOnGPU_ = true;
      auto input = input_ ? input_->getProduct<HeterogeneousDevice::kGPUMock>() : 0U;

      auto ret = std::async(std::launch::async,
                            [this, dur, input,
                             callback = std::move(callback)
                             ](){
                              std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
                              gpuOutput_ = input + streamId_*100 + eventId_;
                              callback();
                            });
      pendingFutures.push_back(std::move(ret));
    }

    auto makeTransfer() const {
      return [this](const unsigned int& src, unsigned int& dst) {
        edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " copying to CPU";
        dst = src;
      };
    }

    bool ranOnGPU() const { return ranOnGPU_; }
    unsigned int getOutput() const { return output_; }
    unsigned int getGPUOutput() const { return gpuOutput_; }

  private:
    // input
    const OutputType *input_ = nullptr;
    unsigned int eventId_ = 0;
    unsigned int streamId_ = 0;

    bool ranOnGPU_ = false;

    // simulating GPU memory
    unsigned int gpuOutput_;

    // output
    unsigned int output_;
  };
}

class TestAcceleratorServiceProducerGPUMock: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestAcceleratorServiceProducerGPUMock(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducerGPUMock() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTask) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  std::string label_;
  AcceleratorService::Token accToken_;

  edm::EDGetTokenT<HeterogeneousProduct> srcToken_;

  TestAlgo algo_;
};


TestAcceleratorServiceProducerGPUMock::TestAcceleratorServiceProducerGPUMock(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  accToken_(edm::Service<AcceleratorService>()->book())
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumes<HeterogeneousProduct>(srcTag);
  }

  produces<HeterogeneousProduct>();
}

void TestAcceleratorServiceProducerGPUMock::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  const OutputType *input = nullptr;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<HeterogeneousProduct> hin;
    iEvent.getByToken(srcToken_, hin);
    input = &(hin->get<OutputType>());
  }

  algo_.setInput(input, iEvent.id().event(), iEvent.streamID());

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "TestAcceleratorServiceProducerGPUMock::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " input " << input;
  edm::Service<AcceleratorService> acc;
  acc->schedule(accToken_, iEvent.streamID(), std::move(waitingTaskHolder), input,
                accelerator::algoGPUMock(&algo_),
                accelerator::algoCPU(&algo_)
                );
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "TestAcceleratorServiceProducerGPUMock::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
}

void TestAcceleratorServiceProducerGPUMock::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "TestAcceleratorServiceProducerGPUMock::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
  std::unique_ptr<HeterogeneousProduct> ret;
  unsigned int value = 0;
  if(algo_.ranOnGPU()) {
    ret = std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::gpuMockProduct(algo_.getGPUOutput()), algo_.makeTransfer()));
    value = ret->get<OutputType>().getProduct<HeterogeneousDevice::kGPUMock>();
  }
  else {
    ret = std::make_unique<HeterogeneousProduct>(OutputType(heterogeneous::cpuProduct(algo_.getOutput())));
    value = ret->get<OutputType>().getProduct<HeterogeneousDevice::kCPU>();
  }

  edm::LogPrint("TestAcceleratorServiceProducerGPUMock") << "TestAcceleratorServiceProducerGPUMock::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " result " << value;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducerGPUMock::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceProducerGPUMock", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducerGPUMock);
