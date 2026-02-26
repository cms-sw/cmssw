#include <catch2/catch_all.hpp>
#include <set>
#include <thread>
#include <iostream>
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

namespace {
  class CustomDeleter {
  public:
    CustomDeleter() = default;
    explicit CustomDeleter(int val) : expectedValue_{val} {}

    void operator()(int* obj) {
      REQUIRE(obj != nullptr);
      REQUIRE(*obj == expectedValue_);
      delete obj;
    }

  private:
    int expectedValue_ = -1;
  };
}  // namespace

TEST_CASE("ReusableObjectHolder", "[ReusableObjectHolder]") {
  SECTION("Construction") {
    {
      edm::ReusableObjectHolder<int> intHolder;
      auto p = intHolder.tryToGet();
      REQUIRE(p.get() == 0);

      intHolder.add(std::make_unique<int>(1));
      p = intHolder.tryToGet();
      REQUIRE(p.get() != 0);
      REQUIRE(*p == 1);

      auto p2 = intHolder.tryToGet();
      REQUIRE(p2.get() == 0);
    }
    {
      edm::ReusableObjectHolder<int> intHolder2;
      auto p3 = intHolder2.makeOrGet([]() -> int* { return new int(1); });
      REQUIRE(p3.get() != 0);
      REQUIRE(*p3 == 1);

      auto p4 = intHolder2.tryToGet();
      REQUIRE(p4.get() == 0);
    }
    {
      edm::ReusableObjectHolder<int> intHolder3;
      auto p5 = intHolder3.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
      REQUIRE(p5.get() != 0);
      REQUIRE(*p5 == 0);

      auto p6 = intHolder3.tryToGet();
      REQUIRE(p6.get() == 0);
    }
    {
      edm::ReusableObjectHolder<int, CustomDeleter> intHolder4;
      auto p7 =
          intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{1}}; });
      REQUIRE(p7.get() != nullptr);
      REQUIRE(*p7 == 1);

      auto p8 = intHolder4.tryToGet();
      REQUIRE(p8.get() == nullptr);
    }
  }

  SECTION("Deletion") {
    {
      edm::ReusableObjectHolder<int> intHolder;
      intHolder.add(std::make_unique<int>(1));
      {
        auto p = intHolder.tryToGet();
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 1);

        auto p2 = intHolder.tryToGet();
        REQUIRE(p2.get() == 0);

        *p = 2;
      }
      {
        auto p = intHolder.tryToGet();
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 2);

        auto p2 = intHolder.tryToGet();
        REQUIRE(p2.get() == 0);
      }
    }
    {
      edm::ReusableObjectHolder<int> intHolder2;
      {
        auto p = intHolder2.makeOrGet([]() -> int* { return new int(1); });
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 1);
        *p = 2;

        auto p2 = intHolder2.tryToGet();
        REQUIRE(p2.get() == 0);
      }
      {
        auto p = intHolder2.makeOrGet([]() -> int* { return new int(1); });
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 2);

        auto p2 = intHolder2.tryToGet();
        REQUIRE(p2.get() == 0);

        auto p3 = intHolder2.makeOrGet([]() -> int* { return new int(1); });
        REQUIRE(p3.get() != 0);
        REQUIRE(*p3 == 1);
      }
    }
    {
      edm::ReusableObjectHolder<int> intHolder3;
      int* address = 0;
      {
        auto p = intHolder3.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 0);
        address = p.get();
        *p = 2;

        auto p2 = intHolder3.tryToGet();
        REQUIRE(p2.get() == 0);
      }
      {
        auto p = intHolder3.makeOrGetAndClear([]() -> int* { return new int(1); }, [](int* iV) { *iV = 0; });
        REQUIRE(p.get() != 0);
        REQUIRE(*p == 0);
        REQUIRE(address == p.get());
      }
    }

    {
      edm::ReusableObjectHolder<int, CustomDeleter> intHolder4;
      {
        auto p =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE(*p == 1);
        *p = 2;

        auto p2 = intHolder4.tryToGet();
        REQUIRE(p2.get() == nullptr);
      }
      {
        auto p =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE(*p == 2);

        auto p2 = intHolder4.tryToGet();
        REQUIRE(p2.get() == nullptr);
      }
      {
        auto p =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{1}, CustomDeleter{2}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE(*p == 2);

        {
          auto p3 =
              intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{3}, CustomDeleter{3}}; });
          REQUIRE(p.get() != nullptr);
          REQUIRE(*p3 == 3);

          auto p4 =
              intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{4}, CustomDeleter{4}}; });
          REQUIRE(p.get() != nullptr);
          REQUIRE(*p4 == 4);
        }

        auto p34 =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{3}, CustomDeleter{3}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE((*p34 == 3 or *p34 == 4));

        auto p43 =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{4}, CustomDeleter{4}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE((*p43 == 3 or *p43 == 4));
        REQUIRE(*p34 != *p43);

        auto p5 =
            intHolder4.makeOrGet([]() { return std::unique_ptr<int, CustomDeleter>{new int{5}, CustomDeleter{5}}; });
        REQUIRE(p.get() != nullptr);
        REQUIRE(*p5 == 5);
      }
    }
  }

  SECTION("SimultaneousUse") {
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

    std::cout << " # seen: " << t1ItemsSeen.size() << " " << t2ItemsSeen.size() << std::endl;
    REQUIRE((t1ItemsSeen.size() > 0 && t1ItemsSeen.size() < 3));
    REQUIRE((t2ItemsSeen.size() > 0 && t2ItemsSeen.size() < 3));
  }
}
