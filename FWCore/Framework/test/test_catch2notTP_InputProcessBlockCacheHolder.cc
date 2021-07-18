#include "catch.hpp"

#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"
#include "FWCore/Framework/interface/global/implementors.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <iostream>

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace edmtest {

  class TestProcessBlockCacheA {
    int x_;
  };

  class TestProcessBlockCacheB {
    int x_;
  };

  class TestProcessBlockCacheC {
    int x_;
  };

  class TestProcessBlockCacheD {
    int x_;
  };

  class TestInputBlockCacheHolder0
      : public edm::global::impl::InputProcessBlockCacheHolder<edm::global::EDProducerBase> {
  public:
    bool wantsProcessBlocks() const override { return true; }
    bool wantsInputProcessBlocks() const override { return true; }
    bool wantsGlobalRuns() const override { return true; }
    bool wantsGlobalLuminosityBlocks() const override { return true; }
    bool wantsStreamRuns() const override { return true; }
    bool wantsStreamLuminosityBlocks() const override { return true; }
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
  };

  class TestInputBlockCacheHolder1
      : public edm::global::impl::InputProcessBlockCacheHolder<edm::global::EDProducerBase, TestProcessBlockCacheA> {
  public:
    bool wantsProcessBlocks() const override { return true; }
    bool wantsInputProcessBlocks() const override { return true; }
    bool wantsGlobalRuns() const override { return true; }
    bool wantsGlobalLuminosityBlocks() const override { return true; }
    bool wantsStreamRuns() const override { return true; }
    bool wantsStreamLuminosityBlocks() const override { return true; }
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
  };

  class TestInputBlockCacheHolder2 : public edm::global::impl::InputProcessBlockCacheHolder<edm::global::EDProducerBase,
                                                                                            TestProcessBlockCacheA,
                                                                                            TestProcessBlockCacheB> {
  public:
    bool wantsProcessBlocks() const override { return true; }
    bool wantsInputProcessBlocks() const override { return true; }
    bool wantsGlobalRuns() const override { return true; }
    bool wantsGlobalLuminosityBlocks() const override { return true; }
    bool wantsStreamRuns() const override { return true; }
    bool wantsStreamLuminosityBlocks() const override { return true; }
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
  };

  class TestInputBlockCacheHolder3 : public edm::global::impl::InputProcessBlockCacheHolder<edm::global::EDProducerBase,
                                                                                            TestProcessBlockCacheA,
                                                                                            TestProcessBlockCacheB,
                                                                                            TestProcessBlockCacheC> {
  public:
    bool wantsProcessBlocks() const override { return true; }
    bool wantsInputProcessBlocks() const override { return true; }
    bool wantsGlobalRuns() const override { return true; }
    bool wantsGlobalLuminosityBlocks() const override { return true; }
    bool wantsStreamRuns() const override { return true; }
    bool wantsStreamLuminosityBlocks() const override { return true; }
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}
  };
}  // namespace edmtest

TEST_CASE("test InputProcessBlockCacheHolder", "[InputProcessBlockCacheHolder]") {
  SECTION("test countTypeInParameterPack") {
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA>() == 0);
    STATIC_REQUIRE(
        edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA, edmtest::TestProcessBlockCacheA>() == 1);
    STATIC_REQUIRE(
        edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA, edmtest::TestProcessBlockCacheB>() == 0);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB>() == 1);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheA>() == 1);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA>() == 2);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheB>() == 0);

    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheA>() == 2);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheC>() == 1);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA>() == 3);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheC>() == 2);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheA>() == 1);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheC>() == 0);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheA>() == 2);
    STATIC_REQUIRE(edm::impl::countTypeInParameterPack<edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheB,
                                                       edmtest::TestProcessBlockCacheA,
                                                       edmtest::TestProcessBlockCacheC>() == 1);

    // The following should not compile and I manually verified it doesn't
    // REQUIRE(edm::impl::countTypeInParameterPack<>() == 0);
  }

  SECTION("test indexInputProcessBlockCache") {
    STATIC_REQUIRE(
        edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheA, edmtest::TestProcessBlockCacheA>() ==
        0);
    STATIC_REQUIRE(edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheA,
                                                          edmtest::TestProcessBlockCacheA,
                                                          edmtest::TestProcessBlockCacheB>() == 0);
    STATIC_REQUIRE(edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheA,
                                                          edmtest::TestProcessBlockCacheB,
                                                          edmtest::TestProcessBlockCacheA>() == 1);
    STATIC_REQUIRE(edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheD,
                                                          edmtest::TestProcessBlockCacheA,
                                                          edmtest::TestProcessBlockCacheB,
                                                          edmtest::TestProcessBlockCacheC,
                                                          edmtest::TestProcessBlockCacheD>() == 3);
    // The following fails compilation if uncommented, tested manually
    // REQUIRE(edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheD, edmtest::TestProcessBlockCacheA, edmtest::TestProcessBlockCacheB, edmtest::TestProcessBlockCacheC>() == 3);
    // REQUIRE(edm::impl::indexInputProcessBlockCache<edmtest::TestProcessBlockCacheD>() == 3);
  }

  SECTION("test constructor") {
    edmtest::TestInputBlockCacheHolder0 holder0;
    edmtest::TestInputBlockCacheHolder1 holder1;
    edmtest::TestInputBlockCacheHolder2 holder2;
    edmtest::TestInputBlockCacheHolder3 holder3;
  }
}
