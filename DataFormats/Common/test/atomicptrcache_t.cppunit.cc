#include "cppunit/extensions/HelperMacros.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"

#include <vector>

class testAtomicPtrCache : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAtomicPtrCache);
  CPPUNIT_TEST(check);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void check();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testAtomicPtrCache);

typedef std::vector<int> Vec;
void
testAtomicPtrCache::check()
{
  using namespace edm;

  const Vec values = {1,2,3};
  
  {
    AtomicPtrCache<Vec> cache;
    CPPUNIT_ASSERT(false == cache.isSet());
    CPPUNIT_ASSERT(nullptr == cache.operator->());
    CPPUNIT_ASSERT(nullptr == cache.load());
    std::unique_ptr<Vec> p{ new Vec{values} };

    cache.set(std::move(p));
    CPPUNIT_ASSERT(true == cache.isSet());
    CPPUNIT_ASSERT(cache->size() == values.size());
  }
  
  {
    AtomicPtrCache<Vec> cache{new Vec{values}};
    CPPUNIT_ASSERT(true == cache.isSet());
    CPPUNIT_ASSERT(cache->size() == values.size());
  }
  
  {
    AtomicPtrCache<Vec> cache;
    CPPUNIT_ASSERT(false == cache.isSet());

    AtomicPtrCache<Vec> cache2{new Vec{values}};
    
    cache = cache2;
    
    CPPUNIT_ASSERT(true == cache.isSet());
    CPPUNIT_ASSERT(cache->size() == values.size());
    CPPUNIT_ASSERT(true == cache2.isSet());
    CPPUNIT_ASSERT(cache2->size() == values.size());
    CPPUNIT_ASSERT(cache.load() != cache2.load());
  }

  {
    AtomicPtrCache<Vec> cache2{new Vec{values}};
    AtomicPtrCache<Vec> cache{ cache2 };
    CPPUNIT_ASSERT(true == cache.isSet());
    CPPUNIT_ASSERT(cache->size() == values.size());
    CPPUNIT_ASSERT(true == cache2.isSet());
    CPPUNIT_ASSERT(cache2->size() == values.size());
    CPPUNIT_ASSERT(cache.load() != cache2.load());
  }

}