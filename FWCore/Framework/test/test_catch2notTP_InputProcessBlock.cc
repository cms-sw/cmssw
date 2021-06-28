#include "catch.hpp"

#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/stream/dummy_helpers.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include <memory>
#include <type_traits>

namespace edmtest {

  class TestAnalyzerWithInputProcessBlockAbility
      : public edm::stream::EDAnalyzer<edm::GlobalCache<int>, edm::InputProcessBlockCache<int, unsigned int, int>> {};

  class TestAnalyzerWithoutInputProcessBlockAbility : public edm::stream::EDAnalyzer<edm::GlobalCache<int>> {};

  class TestFilterWithInputProcessBlockAbility
      : public edm::stream::EDFilter<edm::GlobalCache<int>, edm::InputProcessBlockCache<int, unsigned int, int>> {};

  class TestFilterWithoutInputProcessBlockAbility : public edm::stream::EDFilter<edm::GlobalCache<int>> {};

  class TestProducerWithInputProcessBlockAbility
      : public edm::stream::EDProducer<edm::GlobalCache<int>, edm::InputProcessBlockCache<int, unsigned int, int>> {};

  class TestProducerWithoutInputProcessBlockAbility : public edm::stream::EDProducer<edm::GlobalCache<int>> {};

}  // namespace edmtest

TEST_CASE("test InputProcessBlock", "[InputProcessBlock]") {
  SECTION("test HasAbility") {
    STATIC_REQUIRE(edmtest::TestAnalyzerWithInputProcessBlockAbility::HasAbility::kInputProcessBlockCache);
    STATIC_REQUIRE(!edmtest::TestAnalyzerWithoutInputProcessBlockAbility::HasAbility::kInputProcessBlockCache);
  }
  SECTION("test type aliases") {
    STATIC_REQUIRE(std::is_same<edmtest::TestAnalyzerWithInputProcessBlockAbility::InputProcessBlockCache,
                                edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>());
    STATIC_REQUIRE(std::is_same<edmtest::TestAnalyzerWithoutInputProcessBlockAbility::InputProcessBlockCache, void>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestAnalyzerWithInputProcessBlockAbility::InputProcessBlockCache>::type,
                                std::unique_ptr<edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestAnalyzerWithoutInputProcessBlockAbility::InputProcessBlockCache>::type,
                                edm::stream::impl::dummy_ptr>());

    STATIC_REQUIRE(std::is_same<edmtest::TestFilterWithInputProcessBlockAbility::InputProcessBlockCache,
                                edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>());
    STATIC_REQUIRE(std::is_same<edmtest::TestFilterWithoutInputProcessBlockAbility::InputProcessBlockCache, void>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestFilterWithInputProcessBlockAbility::InputProcessBlockCache>::type,
                                std::unique_ptr<edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestFilterWithoutInputProcessBlockAbility::InputProcessBlockCache>::type,
                                edm::stream::impl::dummy_ptr>());

    STATIC_REQUIRE(std::is_same<edmtest::TestProducerWithInputProcessBlockAbility::InputProcessBlockCache,
                                edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>());
    STATIC_REQUIRE(std::is_same<edmtest::TestProducerWithoutInputProcessBlockAbility::InputProcessBlockCache, void>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestProducerWithInputProcessBlockAbility::InputProcessBlockCache>::type,
                                std::unique_ptr<edm::impl::InputProcessBlockCacheImpl<int, unsigned int, int>>>());
    STATIC_REQUIRE(std::is_same<edm::stream::impl::choose_unique_ptr<
                                    edmtest::TestProducerWithoutInputProcessBlockAbility::InputProcessBlockCache>::type,
                                edm::stream::impl::dummy_ptr>());
  }
}
