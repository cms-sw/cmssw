#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/AtomicPtrCache.h"

#include <vector>

typedef std::vector<int> Vec;

TEST_CASE("test AtomicPtrCache", "[AtomicPtrCache]") {
  using namespace edm;

  const Vec values = {1, 2, 3};

  {
    AtomicPtrCache<Vec> cache;
    REQUIRE(false == cache.isSet());
    REQUIRE(nullptr == cache.operator->());
    REQUIRE(nullptr == cache.load());

    cache.set(std::make_unique<Vec>(values));
    REQUIRE(true == cache.isSet());
    REQUIRE(cache->size() == values.size());
    REQUIRE((*cache).size() == values.size());
    REQUIRE(cache.load()->size() == values.size());

    {
      //test const functions
      const AtomicPtrCache<Vec>& constCache = cache;

      REQUIRE(true == constCache.isSet());
      REQUIRE(constCache->size() == values.size());
      REQUIRE((*constCache).size() == values.size());
      REQUIRE(constCache.load()->size() == values.size());
    }

    cache.reset();
    REQUIRE(false == cache.isSet());
  }

  {
    AtomicPtrCache<Vec> cache{new Vec{values}};
    REQUIRE(true == cache.isSet());
    REQUIRE(cache->size() == values.size());
  }

  {
    AtomicPtrCache<Vec> cache;
    REQUIRE(false == cache.isSet());

    AtomicPtrCache<Vec> cache2{new Vec{values}};

    cache = cache2;

    REQUIRE(true == cache.isSet());
    REQUIRE(cache->size() == values.size());
    REQUIRE(true == cache2.isSet());
    REQUIRE(cache2->size() == values.size());
    REQUIRE(cache.load() != cache2.load());
  }

  {
    AtomicPtrCache<Vec> cache2{new Vec{values}};
    AtomicPtrCache<Vec> cache{cache2};
    REQUIRE(true == cache.isSet());
    REQUIRE(cache->size() == values.size());
    REQUIRE(true == cache2.isSet());
    REQUIRE(cache2->size() == values.size());
    REQUIRE(cache.load() != cache2.load());
  }
}
