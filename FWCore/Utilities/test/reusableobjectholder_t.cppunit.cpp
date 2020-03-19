#include <set>
#include <thread>
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include <cppunit/extensions/HelperMacros.h>

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)
#define CXX_THREAD_AVAILABLE
#endif

class reusableobjectholder_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(reusableobjectholder_test);
  CPPUNIT_TEST(testConstruction);
  CPPUNIT_TEST(testDeletion);
  CPPUNIT_TEST(testSimultaneousUse);
  CPPUNIT_TEST_SUITE_END();

public:
  void testConstruction();
  void testDeletion();
  void testSimultaneousUse();

  void setUp() {}
  void tearDown() {}
};

namespace {
  class CustomDeleter {
  public:
    CustomDeleter() = default;
    explicit CustomDeleter(int val) : expectedValue_{val} {}

    void operator()(int* obj) {
      CPPUNIT_ASSERT(obj != nullptr);
      CPPUNIT_ASSERT(*obj == expectedValue_);
      delete obj;
    }

  private:
    int expectedValue_ = -1;
  };
}  // namespace

void reusableobjectholder_test::testConstruction() {
  {
    edm::ReusableObjectHolder<int> intHolder;
    auto p = intHolder.tryToGet();
    CPPUNIT_ASSERT(p.get() == 0);

    intHolder.add(std::make_unique<int>(1));
    p = intHolder.tryToGet();
    CPPUNIT_ASSERT(p.get() != 0);
    CPPUNIT_ASSERT(*p == 1);

    auto p2 = intHolder.tryToGet();
    CPPUNIT_ASSERT(p2.get() == 0);
  }
  {
    edm::ReusableObjectHolder<int> intHolder;
    auto p = intHolder.makeOrGet([]() -> int* { return new int(1); });
    CPPUNIT_ASSERT(p.get() != 0);
    CPPUNIT_ASSERT(*p == 1);

    auto p2 = intHolder.tryToGet();
    CPPUNIT_ASSERT(p2.get() == 0);
  }
  {
    edm::ReusableObjectHolder<int> intHolder;
    auto p = intHolder.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
    CPPUNIT_ASSERT(p.get() != 0);
    CPPUNIT_ASSERT(*p == 0);

    auto p2 = intHolder.tryToGet();
    CPPUNIT_ASSERT(p2.get() == 0);
  }
  {
    edm::ReusableObjectHolder<int, CustomDeleter> intHolder;
    auto p = intHolder.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{1}}; });
    CPPUNIT_ASSERT(p.get() != nullptr);
    CPPUNIT_ASSERT(*p == 1);

    auto p2 = intHolder.tryToGet();
    CPPUNIT_ASSERT(p2.get() == nullptr);
  }
}

void reusableobjectholder_test::testDeletion() {
  //should also test that makeOrGetAndClear actually does a clear on returned objects
  {
    edm::ReusableObjectHolder<int> intHolder;
    intHolder.add(std::make_unique<int>(1));
    {
      auto p = intHolder.tryToGet();
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 1);

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == 0);

      *p = 2;
    }
    {
      auto p = intHolder.tryToGet();
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 2);

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == 0);
    }
  }

  {
    edm::ReusableObjectHolder<int> intHolder;
    {
      auto p = intHolder.makeOrGet([]() -> int* { return new int(1); });
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 1);
      *p = 2;

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == 0);
    }
    {
      auto p = intHolder.makeOrGet([]() -> int* { return new int(1); });
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 2);

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == 0);

      auto p3 = intHolder.makeOrGet([]() -> int* { return new int(1); });
      CPPUNIT_ASSERT(p3.get() != 0);
      CPPUNIT_ASSERT(*p3 == 1);
    }
  }

  {
    edm::ReusableObjectHolder<int> intHolder;
    int* address = 0;
    {
      auto p = intHolder.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 0);
      address = p.get();
      *p = 2;

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == 0);
    }
    {
      auto p = intHolder.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
      CPPUNIT_ASSERT(p.get() != 0);
      CPPUNIT_ASSERT(*p == 0);
      CPPUNIT_ASSERT(address == p.get());
    }
  }

  {
    edm::ReusableObjectHolder<int, CustomDeleter> intHolder;
    {
      auto p = intHolder.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p == 1);
      *p = 2;

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == nullptr);
    }
    {
      auto p = intHolder.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p == 2);

      auto p2 = intHolder.tryToGet();
      CPPUNIT_ASSERT(p2.get() == nullptr);
    }
    {
      auto p = intHolder.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p == 2);

      {
        auto p3 = intHolder.makeOrGet([]() {
          return std::unique_ptr<int, CustomDeleter>{new int{3}, CustomDeleter{3}};
        });
        CPPUNIT_ASSERT(p.get() != nullptr);
        CPPUNIT_ASSERT(*p3 == 3);

        auto p4 = intHolder.makeOrGet([]() {
          return std::unique_ptr<int, CustomDeleter>{new int{4}, CustomDeleter{4}};
        });
        CPPUNIT_ASSERT(p.get() != nullptr);
        CPPUNIT_ASSERT(*p4 == 4);
      }

      auto p34 = intHolder.makeOrGet([]() {
        return std::unique_ptr<int, CustomDeleter>{new int{3}, CustomDeleter{3}};
      });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p34 == 3 or *p34 == 4);

      auto p43 = intHolder.makeOrGet([]() {
        return std::unique_ptr<int, CustomDeleter>{new int{4}, CustomDeleter{4}};
      });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p43 == 3 or *p43 == 4);
      CPPUNIT_ASSERT(*p34 != *p43);

      auto p5 = intHolder.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{5}, CustomDeleter{5}}; });
      CPPUNIT_ASSERT(p.get() != nullptr);
      CPPUNIT_ASSERT(*p5 == 5);
    }
  }
}
void reusableobjectholder_test::testSimultaneousUse() {
#if defined(CXX_THREAD_AVAILABLE)

  std::set<int*> t1ItemsSeen, t2ItemsSeen;
  edm::ReusableObjectHolder<int> intHolder;

  const unsigned int kNGets = 10000000;

  std::thread t1([&]() {
    for (unsigned int i = 0; i < kNGets; ++i) {
      auto p = intHolder.makeOrGet([]() -> int* { return new int(1); });
      t1ItemsSeen.insert(p.get());
    }
  });

  std::thread t2([&]() {
    for (unsigned int i = 0; i < kNGets; ++i) {
      auto p = intHolder.makeOrGet([]() -> int* { return new int(1); });
      t2ItemsSeen.insert(p.get());
    }
  });

  t1.join();
  t2.join();

  CPPUNIT_ASSERT(t1ItemsSeen.size() > 0 && t1ItemsSeen.size() < 3);
  CPPUNIT_ASSERT(t2ItemsSeen.size() > 0 && t2ItemsSeen.size() < 3);
  //std::cout <<" # seen: "<<t1ItemsSeen.size() <<" "<<t2ItemsSeen.size()<<std::endl;
#endif
}

CPPUNIT_TEST_SUITE_REGISTRATION(reusableobjectholder_test);
