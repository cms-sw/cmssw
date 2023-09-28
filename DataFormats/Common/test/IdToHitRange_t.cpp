#include "catch.hpp"
#include "DataFormats/Common/interface/IdToHitRange.h"

namespace {
  struct Hit {
    Hit(int iValue) : value_(iValue) {}
    int value_ = -1;
  };
}  // namespace

TEST_CASE("test IdToHitRange", "[IdToHitRange]") {
  SECTION("empty") {
    edm::IdToHitRange<int, Hit> hr{};

    REQUIRE(0 == hr.size());
    REQUIRE(hr.begin() == hr.end());
    REQUIRE(hr.id_size() == 0);
    REQUIRE(hr.id_begin() == hr.id_end());
    REQUIRE(hr.ids().empty());
    REQUIRE(hr.get(0) == std::make_pair(hr.end(), hr.end()));
  }

  SECTION("one entry") {
    SECTION("with one hit") {
      edm::IdToHitRange<int, Hit> hr{};
      std::array<Hit, 1> a = {{0}};
      hr.put(0, a.begin(), a.end());

      REQUIRE(1 == hr.size());
      REQUIRE(hr.begin() != hr.end());
      REQUIRE(hr.end() - hr.begin() == 1);
      REQUIRE(hr.get(0) == std::make_pair(hr.begin(), hr.end()));
      REQUIRE(hr.get(0).first->value_ == 0);
      REQUIRE(hr.get(1) == std::make_pair(hr.end(), hr.end()));
      REQUIRE(hr.id_size() == 1);
      REQUIRE(hr.id_begin() != hr.id_end());
      REQUIRE(hr.id_end() - hr.id_begin() == 1);
      REQUIRE(*hr.id_begin() == 0);
      REQUIRE(hr.ids().size() == 1);
    }

    SECTION("with two hit") {
      edm::IdToHitRange<int, Hit> hr{};
      std::array<Hit, 2> a = {{{0}, {0}}};
      hr.put(0, a.begin(), a.end());

      REQUIRE(2 == hr.size());
      REQUIRE(hr.begin() != hr.end());
      REQUIRE(hr.end() - hr.begin() == 2);
      REQUIRE(hr.get(0) == std::make_pair(hr.begin(), hr.end()));
      REQUIRE(hr.get(0).first->value_ == 0);
      REQUIRE(hr.get(1) == std::make_pair(hr.end(), hr.end()));
      REQUIRE(hr.id_size() == 1);
      REQUIRE(hr.id_begin() != hr.id_end());
      REQUIRE(hr.id_end() - hr.id_begin() == 1);
      REQUIRE(*hr.id_begin() == 0);
      REQUIRE(hr.ids().size() == 1);
    }
  }

  SECTION("two entries") {
    SECTION("consecutive IDs") {
      SECTION("added in order") {
        edm::IdToHitRange<int, Hit> hr{};
        std::array<Hit, 1> a = {{0}};
        hr.put(0, a.begin(), a.end());
        std::array<Hit, 2> b = {{{1}, {1}}};
        hr.put(1, b.begin(), b.end());

        REQUIRE(3 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 3);
        REQUIRE(hr.get(0) == std::make_pair(hr.begin(), hr.begin() + 1));
        REQUIRE(hr.get(0).first->value_ == 0);
        auto get_1 = hr.get(1);
        REQUIRE(get_1 == std::make_pair(hr.begin() + 1, hr.end()));
        REQUIRE(get_1.second - get_1.first == 2);
        REQUIRE(get_1.first->value_ == 1);
        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(2) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.id_size() == 2);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 2);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 1);
        REQUIRE(hr.ids().size() == 2);
      }

      SECTION("added out of order") {
        edm::IdToHitRange<int, Hit> hr{};
        std::array<Hit, 2> b = {{{1}, {1}}};
        hr.put(1, b.begin(), b.end());
        std::array<Hit, 1> a = {{0}};
        hr.put(0, a.begin(), a.end());

        REQUIRE(hr.id_size() == 2);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 2);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 1);
        REQUIRE(hr.ids().size() == 2);

        REQUIRE(3 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 3);
        auto get_0 = hr.get(0);
        REQUIRE(get_0.first != hr.end());
        REQUIRE(get_0.first->value_ == 0);
        REQUIRE(get_0.second - get_0.first == 1);
        REQUIRE(hr.get(0) == std::make_pair(hr.begin() + 2, hr.end()));
        REQUIRE(hr.get(0).first->value_ == 0);
        auto get_1 = hr.get(1);
        REQUIRE(get_1 == std::make_pair(hr.begin(), hr.begin() + 2));
        REQUIRE(get_1.second - get_1.first == 2);
        REQUIRE(get_1.first->value_ == 1);
        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(2) == std::make_pair(hr.end(), hr.end()));

        SECTION("post_insert") {
          hr.post_insert();

          REQUIRE(3 == hr.size());
          REQUIRE(hr.begin() != hr.end());
          REQUIRE(hr.end() - hr.begin() == 3);
          REQUIRE(hr.get(0) == std::make_pair(hr.begin(), hr.begin() + 1));
          REQUIRE(hr.get(0).first->value_ == 0);
          auto get_1 = hr.get(1);
          REQUIRE(get_1 == std::make_pair(hr.begin() + 1, hr.end()));
          REQUIRE(get_1.second - get_1.first == 2);
          REQUIRE(get_1.first->value_ == 1);
          REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
          REQUIRE(hr.get(2) == std::make_pair(hr.end(), hr.end()));
          REQUIRE(hr.id_size() == 2);
          REQUIRE(hr.id_begin() != hr.id_end());
          REQUIRE(hr.id_end() - hr.id_begin() == 2);
          REQUIRE(*hr.id_begin() == 0);
          REQUIRE(*(hr.id_begin() + 1) == 1);
          REQUIRE(hr.ids().size() == 2);
        }
      }
    }

    SECTION("non-consecutive IDs") {
      SECTION("added in order") {
        edm::IdToHitRange<int, Hit> hr{};
        std::array<Hit, 1> a = {{0}};
        hr.put(0, a.begin(), a.end());
        std::array<Hit, 2> b = {{{2}, {2}}};
        hr.put(2, b.begin(), b.end());

        REQUIRE(3 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 3);
        REQUIRE(hr.get(0) == std::make_pair(hr.begin(), hr.begin() + 1));
        REQUIRE(hr.get(0).first->value_ == 0);
        auto get_2 = hr.get(2);
        REQUIRE(get_2 == std::make_pair(hr.begin() + 1, hr.end()));
        REQUIRE(get_2.second - get_2.first == 2);
        REQUIRE(get_2.first->value_ == 2);
        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.id_size() == 2);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 2);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 2);
        REQUIRE(hr.ids().size() == 2);
      }

      SECTION("added out of order") {
        edm::IdToHitRange<int, Hit> hr{};
        std::array<Hit, 2> b = {{{2}, {2}}};
        hr.put(2, b.begin(), b.end());
        std::array<Hit, 1> a = {{0}};
        hr.put(0, a.begin(), a.end());

        REQUIRE(hr.id_size() == 2);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 2);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 2);
        REQUIRE(hr.ids().size() == 2);

        REQUIRE(3 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 3);
        auto get_0 = hr.get(0);
        REQUIRE(get_0.first != hr.end());
        REQUIRE(get_0.first->value_ == 0);
        REQUIRE(get_0.second - get_0.first == 1);
        REQUIRE(hr.get(0) == std::make_pair(hr.begin() + 2, hr.end()));
        REQUIRE(hr.get(0).first->value_ == 0);
        auto get_2 = hr.get(2);
        REQUIRE(get_2 == std::make_pair(hr.begin(), hr.begin() + 2));
        REQUIRE(get_2.second - get_2.first == 2);
        REQUIRE(get_2.first->value_ == 2);
        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));
      }
    }
  }
  SECTION("three entries") {
    SECTION("first two in order, third out") {
      SECTION("last added ID at beginning") {
        edm::IdToHitRange<int, Hit> hr{};
        {
          std::array<Hit, 2> b = {{{1}, {1}}};
          hr.put(1, b.begin(), b.end());
        }
        {
          std::array<Hit, 3> c = {{{2}, {2}, {2}}};
          hr.put(2, c.begin(), c.end());
        }
        {
          std::array<Hit, 1> a = {{0}};
          hr.put(0, a.begin(), a.end());
        }
        REQUIRE(hr.id_size() == 3);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 3);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 1);
        REQUIRE(*(hr.id_begin() + 2) == 2);
        REQUIRE(hr.ids().size() == 3);

        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));

        REQUIRE(6 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 6);
        {
          auto get_0 = hr.get(0);
          REQUIRE(get_0.first != hr.end());
          REQUIRE(get_0.first->value_ == 0);
          REQUIRE(get_0.second - get_0.first == 1);
          REQUIRE(get_0 == std::make_pair(hr.begin() + 5, hr.end()));
        }
        {
          auto get_1 = hr.get(1);
          REQUIRE(get_1.first != hr.end());
          REQUIRE(get_1.first->value_ == 1);
          REQUIRE(get_1.second - get_1.first == 2);
          REQUIRE(get_1 == std::make_pair(hr.begin(), hr.begin() + 2));
        }
        {
          auto get_2 = hr.get(2);
          REQUIRE(get_2.first != hr.end());
          REQUIRE(get_2.first->value_ == 2);
          REQUIRE(get_2.second - get_2.first == 3);
          REQUIRE(get_2 == std::make_pair(hr.begin() + 2, hr.begin() + 5));
        }

        SECTION("post_insert") {
          hr.post_insert();

          REQUIRE(hr.id_size() == 3);
          REQUIRE(hr.id_begin() != hr.id_end());
          REQUIRE(hr.id_end() - hr.id_begin() == 3);
          REQUIRE(*hr.id_begin() == 0);
          REQUIRE(*(hr.id_begin() + 1) == 1);
          REQUIRE(*(hr.id_begin() + 2) == 2);
          REQUIRE(hr.ids().size() == 3);

          REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
          REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));

          REQUIRE(6 == hr.size());
          REQUIRE(hr.begin() != hr.end());
          REQUIRE(hr.end() - hr.begin() == 6);
          {
            auto get_0 = hr.get(0);
            REQUIRE(get_0.first != hr.end());
            REQUIRE(get_0.first->value_ == 0);
            REQUIRE(get_0.second - get_0.first == 1);
            REQUIRE(get_0 == std::make_pair(hr.begin(), hr.begin() + 1));
          }
          {
            auto get_1 = hr.get(1);
            REQUIRE(get_1.first != hr.end());
            REQUIRE(get_1.first->value_ == 1);
            REQUIRE(get_1.second - get_1.first == 2);
            REQUIRE(get_1 == std::make_pair(hr.begin() + 1, hr.begin() + 3));
          }
          {
            auto get_2 = hr.get(2);
            REQUIRE(get_2.first != hr.end());
            REQUIRE(get_2.first->value_ == 2);
            REQUIRE(get_2.second - get_2.first == 3);
            REQUIRE(get_2 == std::make_pair(hr.begin() + 3, hr.end()));
          }
        }
      }
      SECTION("last added ID in middle") {
        edm::IdToHitRange<int, Hit> hr{};
        {
          std::array<Hit, 1> a = {{0}};
          hr.put(0, a.begin(), a.end());
        }
        {
          std::array<Hit, 3> c = {{{2}, {2}, {2}}};
          hr.put(2, c.begin(), c.end());
        }
        {
          std::array<Hit, 2> b = {{{1}, {1}}};
          hr.put(1, b.begin(), b.end());
        }
        REQUIRE(hr.id_size() == 3);
        REQUIRE(hr.id_begin() != hr.id_end());
        REQUIRE(hr.id_end() - hr.id_begin() == 3);
        REQUIRE(*hr.id_begin() == 0);
        REQUIRE(*(hr.id_begin() + 1) == 1);
        REQUIRE(*(hr.id_begin() + 2) == 2);
        REQUIRE(hr.ids().size() == 3);

        REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
        REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));

        REQUIRE(6 == hr.size());
        REQUIRE(hr.begin() != hr.end());
        REQUIRE(hr.end() - hr.begin() == 6);
        {
          auto get_0 = hr.get(0);
          REQUIRE(get_0.first != hr.end());
          REQUIRE(get_0.first->value_ == 0);
          REQUIRE(get_0.second - get_0.first == 1);
          REQUIRE(get_0 == std::make_pair(hr.begin(), hr.begin() + 1));
        }
        {
          auto get_1 = hr.get(1);
          REQUIRE(get_1.first != hr.end());
          REQUIRE(get_1.first->value_ == 1);
          REQUIRE(get_1.second - get_1.first == 2);
          REQUIRE(get_1 == std::make_pair(hr.begin() + 4, hr.end()));
        }
        {
          auto get_2 = hr.get(2);
          REQUIRE(get_2.first != hr.end());
          REQUIRE(get_2.first->value_ == 2);
          REQUIRE(get_2.second - get_2.first == 3);
          REQUIRE(get_2 == std::make_pair(hr.begin() + 1, hr.begin() + 4));
        }

        SECTION("post_insert") {
          hr.post_insert();

          REQUIRE(hr.id_size() == 3);
          REQUIRE(hr.id_begin() != hr.id_end());
          REQUIRE(hr.id_end() - hr.id_begin() == 3);
          REQUIRE(*hr.id_begin() == 0);
          REQUIRE(*(hr.id_begin() + 1) == 1);
          REQUIRE(*(hr.id_begin() + 2) == 2);
          REQUIRE(hr.ids().size() == 3);

          REQUIRE(hr.get(-1) == std::make_pair(hr.end(), hr.end()));
          REQUIRE(hr.get(3) == std::make_pair(hr.end(), hr.end()));

          REQUIRE(6 == hr.size());
          REQUIRE(hr.begin() != hr.end());
          REQUIRE(hr.end() - hr.begin() == 6);
          {
            auto get_0 = hr.get(0);
            REQUIRE(get_0.first != hr.end());
            REQUIRE(get_0.first->value_ == 0);
            REQUIRE(get_0.second - get_0.first == 1);
            REQUIRE(get_0 == std::make_pair(hr.begin(), hr.begin() + 1));
          }
          {
            auto get_1 = hr.get(1);
            REQUIRE(get_1.first != hr.end());
            REQUIRE(get_1.first->value_ == 1);
            REQUIRE(get_1.second - get_1.first == 2);
            REQUIRE(get_1 == std::make_pair(hr.begin() + 1, hr.begin() + 3));
          }
          {
            auto get_2 = hr.get(2);
            REQUIRE(get_2.first != hr.end());
            REQUIRE(get_2.first->value_ == 2);
            REQUIRE(get_2.second - get_2.first == 3);
            REQUIRE(get_2 == std::make_pair(hr.begin() + 3, hr.end()));
          }
        }
      }
    }
  }
  SECTION("get with comparator") {
    edm::IdToHitRange<int, Hit> hr{};
    {
      std::array<Hit, 1> a = {{0}};
      hr.put(0, a.begin(), a.end());
    }
    {
      std::array<Hit, 2> b = {{{1}, {1}}};
      hr.put(1, b.begin(), b.end());
    }
    {
      std::array<Hit, 3> c = {{{2}, {2}, {2}}};
      hr.put(2, c.begin(), c.end());
    }

    SECTION("none before") {
      auto range = hr.get(-2, [](auto i, auto j) { return i / 2 < j / 2; });
      REQUIRE(range.second == range.first);
    }

    SECTION("none after") {
      auto range = hr.get(4, [](auto i, auto j) { return i / 2 < j / 2; });
      REQUIRE(range.second == range.first);
    }

    SECTION("matches") {
      SECTION("match to 0") {
        auto range = hr.get(0, [](auto i, auto j) { return i / 2 < j / 2; });
        REQUIRE(range.second != range.first);
        REQUIRE(range.second - range.first == 3);
        REQUIRE(range.first->value_ == 0);
      }
      SECTION("match to 1") {
        auto range = hr.get(0, [](auto i, auto j) { return i / 2 < j / 2; });
        REQUIRE(range.second != range.first);
        REQUIRE(range.second - range.first == 3);
        REQUIRE(range.first->value_ == 0);
      }
      SECTION("match to 2") {
        auto range = hr.get(2, [](auto i, auto j) { return i / 2 < j / 2; });
        REQUIRE(range.second != range.first);
        REQUIRE(range.second - range.first == 3);
        REQUIRE(range.first->value_ == 2);
      }
      SECTION("match to 3") {
        auto range = hr.get(3, [](auto i, auto j) { return i / 2 < j / 2; });
        REQUIRE(range.second != range.first);
        REQUIRE(range.second - range.first == 3);
        REQUIRE(range.first->value_ == 2);
      }
    }
  }
}
