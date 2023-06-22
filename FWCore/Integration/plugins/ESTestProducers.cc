#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include <memory>
#include <vector>

namespace edmtest {

  class ESTestProducerA : public edm::ESProducer {
  public:
    ESTestProducerA(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataA> produce(ESTestRecordA const&);

  private:
    int value_;
  };

  ESTestProducerA::ESTestProducerA(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataA> ESTestProducerA::produce(ESTestRecordA const& rec) {
    ++value_;
    return std::make_unique<ESTestDataA>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerB : public edm::ESProducer {
  public:
    ESTestProducerB(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataB> produce(ESTestRecordB const&);

  private:
    int value_;
  };

  ESTestProducerB::ESTestProducerB(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataB> ESTestProducerB::produce(ESTestRecordB const& rec) {
    ++value_;
    return std::make_unique<ESTestDataB>(value_);
  }

  // ---------------------------------------------------------------------

  // This class is used to test ESProductHost
  class ESTestProducerBUsingHost : public edm::ESProducer {
  public:
    ESTestProducerBUsingHost(edm::ParameterSet const&);
    // Must use shared_ptr if using ReusableObjectHolder
    std::shared_ptr<ESTestDataB> produce(ESTestRecordB const&);

  private:
    using HostType = edm::ESProductHost<ESTestDataB,
                                        ESTestRecordC,
                                        ESTestRecordD,
                                        ESTestRecordE,
                                        ESTestRecordF,
                                        ESTestRecordG,
                                        ESTestRecordH>;

    edm::ReusableObjectHolder<HostType> holder_;
  };

  ESTestProducerBUsingHost::ESTestProducerBUsingHost(edm::ParameterSet const&) { setWhatProduced(this); }

  std::shared_ptr<ESTestDataB> ESTestProducerBUsingHost::produce(ESTestRecordB const& record) {
    auto host = holder_.makeOrGet([]() { return new HostType(100, 1000); });

    // Test that the numberOfRecordTypes and index functions are working properly
    if (host->numberOfRecordTypes() != 6 || host->index<ESTestRecordC>() != 0 || host->index<ESTestRecordD>() != 1 ||
        host->index<ESTestRecordE>() != 2 || host->index<ESTestRecordF>() != 3 || host->index<ESTestRecordG>() != 4 ||
        host->index<ESTestRecordH>() != 5) {
      throw cms::Exception("TestError") << "Either function numberOfRecordTypes or index returns incorrect value";
    }

    host->ifRecordChanges<ESTestRecordC>(record, [h = host.get()](auto const& rec) { ++h->value(); });

    ++host->value();
    return host;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerC : public edm::ESProducer {
  public:
    ESTestProducerC(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataC> produce(ESTestRecordC const&);

  private:
    int value_;
  };

  ESTestProducerC::ESTestProducerC(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataC> ESTestProducerC::produce(ESTestRecordC const& rec) {
    ++value_;
    return std::make_unique<ESTestDataC>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerD : public edm::ESProducer {
  public:
    ESTestProducerD(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataD> produce(ESTestRecordD const&);

  private:
    int value_;
  };

  ESTestProducerD::ESTestProducerD(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataD> ESTestProducerD::produce(ESTestRecordD const& rec) {
    ++value_;
    return std::make_unique<ESTestDataD>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerE : public edm::ESProducer {
  public:
    ESTestProducerE(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataE> produce(ESTestRecordE const&);

  private:
    int value_;
  };

  ESTestProducerE::ESTestProducerE(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataE> ESTestProducerE::produce(ESTestRecordE const& rec) {
    ++value_;
    return std::make_unique<ESTestDataE>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerF : public edm::ESProducer {
  public:
    ESTestProducerF(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataF> produce(ESTestRecordF const&);

  private:
    int value_;
  };

  ESTestProducerF::ESTestProducerF(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataF> ESTestProducerF::produce(ESTestRecordF const& rec) {
    ++value_;
    return std::make_unique<ESTestDataF>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerG : public edm::ESProducer {
  public:
    ESTestProducerG(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataG> produce(ESTestRecordG const&);

  private:
    int value_;
  };

  ESTestProducerG::ESTestProducerG(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataG> ESTestProducerG::produce(ESTestRecordG const& rec) {
    ++value_;
    return std::make_unique<ESTestDataG>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerH : public edm::ESProducer {
  public:
    ESTestProducerH(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataH> produce(ESTestRecordH const&);

  private:
    int value_;
  };

  ESTestProducerH::ESTestProducerH(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataH> ESTestProducerH::produce(ESTestRecordH const& rec) {
    ++value_;
    return std::make_unique<ESTestDataH>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerI : public edm::ESProducer {
  public:
    ESTestProducerI(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataI> produce(ESTestRecordI const&);

  private:
    int value_;
  };

  ESTestProducerI::ESTestProducerI(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataI> ESTestProducerI::produce(ESTestRecordI const& rec) {
    ++value_;
    return std::make_unique<ESTestDataI>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerJ : public edm::ESProducer {
  public:
    ESTestProducerJ(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataJ> produce(ESTestRecordJ const&);

  private:
    int value_;
  };

  ESTestProducerJ::ESTestProducerJ(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataJ> ESTestProducerJ::produce(ESTestRecordJ const& rec) {
    ++value_;
    return std::make_unique<ESTestDataJ>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerK : public edm::ESProducer {
  public:
    ESTestProducerK(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataK> produce(ESTestRecordK const&);

  private:
    int value_;
  };

  ESTestProducerK::ESTestProducerK(edm::ParameterSet const&) : value_(0) { setWhatProduced(this); }

  std::unique_ptr<ESTestDataK> ESTestProducerK::produce(ESTestRecordK const& rec) {
    ++value_;
    return std::make_unique<ESTestDataK>(value_);
  }

  // ---------------------------------------------------------------------

  class ESTestProducerAZ : public edm::ESProducer {
  public:
    ESTestProducerAZ(edm::ParameterSet const&);
    std::unique_ptr<ESTestDataA> produceA(ESTestRecordA const&);
    std::unique_ptr<ESTestDataZ> produceZ(ESTestRecordZ const&);

  private:
    int valueA_;
    int valueZ_;
  };

  ESTestProducerAZ::ESTestProducerAZ(edm::ParameterSet const&) : valueA_(0), valueZ_(0) {
    setWhatProduced(this, &edmtest::ESTestProducerAZ::produceA, edm::es::Label("foo"));
    setWhatProduced(this, &edmtest::ESTestProducerAZ::produceZ, edm::es::Label("foo"));
  }

  std::unique_ptr<ESTestDataA> ESTestProducerAZ::produceA(ESTestRecordA const& rec) {
    ++valueA_;
    return std::make_unique<ESTestDataA>(valueA_);
  }

  std::unique_ptr<ESTestDataZ> ESTestProducerAZ::produceZ(ESTestRecordZ const& rec) {
    ++valueZ_;
    return std::make_unique<ESTestDataZ>(valueZ_);
  }

  class TestESProductResolverTemplateJ : public edm::eventsetup::ESProductResolverTemplate<ESTestRecordJ, ESTestDataJ> {
  public:
    TestESProductResolverTemplateJ(std::vector<unsigned int> const* expectedCacheIds)
        : testDataJ_(1), expectedCacheIds_(expectedCacheIds) {}

  private:
    const ESTestDataJ* make(const ESTestRecordJ& record, const edm::eventsetup::DataKey& key) override {
      ESTestRecordK recordK = record.getRecord<ESTestRecordK>();
      // Note that this test only reliably works when running with a
      // single IOV at a time and a single stream. This test module
      // should not be configured with expected values in other cases.
      if (index_ < expectedCacheIds_->size() && recordK.cacheIdentifier() != expectedCacheIds_->at(index_)) {
        throw cms::Exception("TestError") << "TestESProductResolverTemplateJ::make, unexpected cacheIdentifier";
      }
      ++index_;
      return &testDataJ_;
    }

    void invalidateCache() override {}
    void const* getAfterPrefetchImpl() const override { return &testDataJ_; }

    ESTestDataJ testDataJ_;
    std::vector<unsigned> const* expectedCacheIds_;
    unsigned int index_ = 0;
  };

  class ESTestESProductResolverProviderJ : public edm::eventsetup::ESProductResolverProvider {
  public:
    ESTestESProductResolverProviderJ(edm::ParameterSet const&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    KeyedResolversVector registerResolvers(const edm::eventsetup::EventSetupRecordKey&, unsigned int iovIndex) override;

    std::vector<std::shared_ptr<TestESProductResolverTemplateJ>> proxies_;
    std::vector<unsigned> expectedCacheIds_;
  };

  ESTestESProductResolverProviderJ::ESTestESProductResolverProviderJ(edm::ParameterSet const& pset)
      : expectedCacheIds_(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedCacheIds")) {
    usingRecord<ESTestRecordJ>();
  }

  void ESTestESProductResolverProviderJ::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    std::vector<unsigned int> emptyDefaultVector;
    desc.addUntracked<std::vector<unsigned int>>("expectedCacheIds", emptyDefaultVector);
    descriptions.addDefault(desc);
  }

  edm::eventsetup::ESProductResolverProvider::KeyedResolversVector ESTestESProductResolverProviderJ::registerResolvers(
      const edm::eventsetup::EventSetupRecordKey& iRecord, unsigned int iovIndex) {
    KeyedResolversVector keyedResolversVector;
    while (iovIndex >= proxies_.size()) {
      proxies_.push_back(std::make_shared<TestESProductResolverTemplateJ>(&expectedCacheIds_));
    }
    edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<ESTestDataJ>(), "");
    keyedResolversVector.emplace_back(dataKey, proxies_[iovIndex]);
    return keyedResolversVector;
  }
}  // namespace edmtest

namespace edm::test {
  namespace other {
    class ESTestProducerA : public edm::ESProducer {
    public:
      ESTestProducerA(edm::ParameterSet const& pset) : value_(pset.getParameter<int>("valueOther")) {
        setWhatProduced(this);
      }
      std::optional<edmtest::ESTestDataA> produce(ESTestRecordA const& rec) {
        ++value_;
        return edmtest::ESTestDataA(value_);
      }

    private:
      int value_;
    };
  }  // namespace other
  namespace cpu {
    class ESTestProducerA : public edm::ESProducer {
    public:
      ESTestProducerA(edm::ParameterSet const& pset) : value_(pset.getParameter<int>("valueCpu")) {
        setWhatProduced(this);
      }
      std::optional<edmtest::ESTestDataA> produce(ESTestRecordA const& rec) {
        ++value_;
        return edmtest::ESTestDataA(value_);
      }

    private:
      int value_;
    };
  }  // namespace cpu
}  // namespace edm::test

using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerA);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerB);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerBUsingHost);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerC);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerD);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerE);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerF);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerG);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerH);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerI);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerJ);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerK);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerAZ);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestESProductResolverProviderJ);
DEFINE_FWK_EVENTSETUP_MODULE(edm::test::other::ESTestProducerA);
DEFINE_FWK_EVENTSETUP_MODULE(edm::test::cpu::ESTestProducerA);
