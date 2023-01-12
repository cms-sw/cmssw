#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "WaitingService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class AcquireIntStreamProducer : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    explicit AcquireIntStreamProducer(edm::ParameterSet const& pset);
    ~AcquireIntStreamProducer() override;
    void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
    void produce(edm::Event&, edm::EventSetup const&) override;

  private:
    std::vector<edm::EDGetTokenT<IntProduct>> m_getTokens;
    edm::EDGetTokenT<IntProduct> m_tokenForProduce;
    test_acquire::Token m_token;
  };

  AcquireIntStreamProducer::AcquireIntStreamProducer(edm::ParameterSet const& pset)
      : m_token{edm::Service<test_acquire::WaitingService>()->getToken()} {
    for (auto const& tag : pset.getParameter<std::vector<edm::InputTag>>("tags")) {
      m_getTokens.emplace_back(consumes<IntProduct>(tag));
    }
    m_tokenForProduce = consumes<IntProduct>(pset.getParameter<edm::InputTag>("produceTag"));
    produces<IntProduct>();
  }

  AcquireIntStreamProducer::~AcquireIntStreamProducer() {}

  void AcquireIntStreamProducer::acquire(edm::Event const& event,
                                         edm::EventSetup const&,
                                         edm::WaitingTaskWithArenaHolder holder) {
    test_acquire::Cache* cache = edm::Service<test_acquire::WaitingService>()->getCache(m_token);
    cache->retrieved().clear();
    cache->processed().clear();

    for (auto const& token : m_getTokens) {
      cache->retrieved().push_back(event.get(token).value);
    }

    edm::Service<test_acquire::WaitingService>()->requestValuesAsync(
        m_token, &cache->retrieved(), &cache->processed(), holder);
  }

  void AcquireIntStreamProducer::produce(edm::Event& event, edm::EventSetup const&) {
    int sum = 0;
    test_acquire::Cache* cache = edm::Service<test_acquire::WaitingService>()->getCache(m_token);
    for (auto v : cache->processed()) {
      sum += v;
    }
    event.put(std::make_unique<IntProduct>(sum));

    // This part is here only for the Parentage test.
    (void)event.get(m_tokenForProduce);
  }
}  // namespace edmtest
using edmtest::AcquireIntStreamProducer;
DEFINE_FWK_MODULE(AcquireIntStreamProducer);
