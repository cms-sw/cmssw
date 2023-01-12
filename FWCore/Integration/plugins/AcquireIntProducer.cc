#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "WaitingServer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <unistd.h>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class AcquireIntProducer : public edm::global::EDProducer<edm::ExternalWork, edm::StreamCache<test_acquire::Cache>> {
  public:
    explicit AcquireIntProducer(edm::ParameterSet const& pset);
    ~AcquireIntProducer() override;

    std::unique_ptr<test_acquire::Cache> beginStream(edm::StreamID) const override;

    void acquire(edm::StreamID,
                 edm::Event const&,
                 edm::EventSetup const&,
                 edm::WaitingTaskWithArenaHolder) const override;

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    void endJob() override;

  private:
    void preallocate(edm::PreallocationConfiguration const&) override;

    std::vector<edm::EDGetTokenT<IntProduct>> m_tokens;
    edm::EDGetTokenT<IntProduct> m_tokenForProduce;
    std::unique_ptr<test_acquire::WaitingServer> m_server;
    const unsigned int m_numberOfStreamsToAccumulate;
    const unsigned int m_secondsToWaitForWork;
  };

  AcquireIntProducer::AcquireIntProducer(edm::ParameterSet const& pset)
      : m_numberOfStreamsToAccumulate(pset.getUntrackedParameter<unsigned int>("streamsToAccumulate", 8)),
        m_secondsToWaitForWork(pset.getUntrackedParameter<unsigned int>("secondsToWaitForWork", 1)) {
    for (auto const& tag : pset.getParameter<std::vector<edm::InputTag>>("tags")) {
      m_tokens.emplace_back(consumes<IntProduct>(tag));
    }
    m_tokenForProduce = consumes<IntProduct>(pset.getParameter<edm::InputTag>("produceTag"));
    produces<IntProduct>();
  }

  AcquireIntProducer::~AcquireIntProducer() {
    if (m_server) {
      m_server->stop();
    }
  }

  void AcquireIntProducer::preallocate(edm::PreallocationConfiguration const& iPrealloc) {
    m_server = std::make_unique<test_acquire::WaitingServer>(
        iPrealloc.numberOfStreams(), m_numberOfStreamsToAccumulate, m_secondsToWaitForWork);
    m_server->start();
  }

  std::unique_ptr<test_acquire::Cache> AcquireIntProducer::beginStream(edm::StreamID) const {
    return std::make_unique<test_acquire::Cache>();
  }

  void AcquireIntProducer::acquire(edm::StreamID streamID,
                                   edm::Event const& event,
                                   edm::EventSetup const&,
                                   edm::WaitingTaskWithArenaHolder holder) const {
    usleep(1000000);

    test_acquire::Cache* streamCacheData = streamCache(streamID);
    streamCacheData->retrieved().clear();
    streamCacheData->processed().clear();

    for (auto const& token : m_tokens) {
      streamCacheData->retrieved().push_back(event.get(token).value);
    }
    m_server->requestValuesAsync(
        streamID.value(), &streamCacheData->retrieved(), &streamCacheData->processed(), holder);
  }

  void AcquireIntProducer::produce(edm::StreamID streamID, edm::Event& event, edm::EventSetup const&) const {
    usleep(1000000);

    int sum = 0;
    for (auto v : streamCache(streamID)->processed()) {
      sum += v;
    }
    event.put(std::make_unique<IntProduct>(sum));

    // This part is here only for the Parentage test.
    (void)event.get(m_tokenForProduce);
  }

  void AcquireIntProducer::endJob() {
    if (m_server) {
      m_server->stop();
    }
    m_server.reset();
  }
}  // namespace edmtest

using edmtest::AcquireIntProducer;
DEFINE_FWK_MODULE(AcquireIntProducer);
