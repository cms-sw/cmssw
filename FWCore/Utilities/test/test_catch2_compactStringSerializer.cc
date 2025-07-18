#include "catch.hpp"

#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/compactStringSerializer.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace cs = edm::compactString;

TEST_CASE("Test edm::compactString serializer", "[edm::compactString]") {
  using namespace std::string_literals;
  SECTION("Empty inputs") {
    SECTION("Serialization") {
      SECTION("Empty string") {
        auto result = cs::serialize(""s);
        CHECK(result.size() == 1);  // one delimiter
        result = cs::serialize("");
        CHECK(result.size() == 1);  // one delimiter
      }

      SECTION("Two empty strings") {
        auto result = cs::serialize(""s, ""s);
        CHECK(result.size() == 2);
        result = cs::serialize(""s, "");
        CHECK(result.size() == 2);
        result = cs::serialize("", ""s);
        CHECK(result.size() == 2);
        result = cs::serialize("", "");
        CHECK(result.size() == 2);
      }

      SECTION("Empty vector of strings and empty string") {
        auto result = cs::serialize(std::vector<std::string>());
        CHECK(result.size() == 1);
        result = cs::serialize(""s, std::vector<std::string>());
        CHECK(result.size() == 2);
        result = cs::serialize(std::vector<std::string>(), "");
        CHECK(result.size() == 2);
        result = cs::serialize(std::vector<std::string>(), std::vector<std::string>());
        CHECK(result.size() == 2);
      }

      SECTION("Empty list and vector of strings and empty string") {
        auto result = cs::serialize(std::list<std::string>());
        CHECK(result.size() == 1);
        result = cs::serialize(""s, std::list<std::string>());
        CHECK(result.size() == 2);
        result = cs::serialize(std::list<std::string>(), "");
        CHECK(result.size() == 2);
        result = cs::serialize(std::list<std::string>(), std::list<std::string>());
        CHECK(result.size() == 2);
        result = cs::serialize(std::vector<std::string>(), std::list<std::string>());
        CHECK(result.size() == 2);
        result = cs::serialize(std::list<std::string>(), std::vector<std::string>());
        CHECK(result.size() == 2);
      }

      SECTION("Vectors of empty strings") {
        auto result = cs::serialize(std::vector<std::string>{""});
        CHECK(result.size() == 2);
        result = cs::serialize(std::vector<std::string>{"", ""});
        CHECK(result.size() == 3);
        result = cs::serialize(std::vector<std::string>{""}, std::vector<std::string>{});
        CHECK(result.size() == 3);
        result = cs::serialize(std::vector<std::string>{"", ""}, std::vector<std::string>{});
        CHECK(result.size() == 4);
        result = cs::serialize(std::vector<std::string>{""}, std::vector<std::string>{""});
        CHECK(result.size() == 4);
        result = cs::serialize(std::vector<std::string>{"", ""}, std::vector<std::string>{""});
        CHECK(result.size() == 5);
        result = cs::serialize(std::vector<std::string>{""}, std::vector<std::string>{"", ""});
        CHECK(result.size() == 5);
        result = cs::serialize(std::vector<std::string>{"", ""}, std::vector<std::string>{"", ""});
        CHECK(result.size() == 6);
      }
    }

    SECTION("Serialization and deserialization") {
      SECTION("Empty string") {
        std::string res;
        auto ret = cs::deserialize(cs::serialize(""), res);
        CHECK(ret == 1);
        CHECK(res.empty());
      }

      SECTION("Two empty strings") {
        std::string res1, res2;
        auto ret = cs::deserialize(cs::serialize("", ""), res1, res2);
        CHECK(ret == 2);
        CHECK(res1.empty());
        CHECK(res2.empty());
      }

      SECTION("Empty vector") {
        std::vector<std::string> res;
        auto ret = cs::deserialize(cs::serialize(std::vector<std::string>()), std::back_inserter(res));
        CHECK(ret == 1);
        CHECK(res.empty());
      }

      SECTION("Two empty vectors") {
        std::vector<std::string> res1, res2;
        auto ret = cs::deserialize(cs::serialize(std::vector<std::string>(), std::vector<std::string>()),
                                   std::back_inserter(res1),
                                   std::back_inserter(res2));
        CHECK(ret == 2);
        CHECK(res1.empty());
        CHECK(res2.empty());
      }

      SECTION("Mixture") {
        std::string res1;
        std::vector<std::string> res2;
        std::list<std::string> res3;
        auto ret = cs::deserialize(cs::serialize("", std::vector<std::string>(), std::list<std::string>()),
                                   res1,
                                   std::back_inserter(res2),
                                   std::back_inserter(res3));
        CHECK(ret == 3);
        CHECK(res1.empty());
        CHECK(res2.empty());
        CHECK(res3.empty());

        ret = cs::deserialize(cs::serialize(std::vector<std::string>(), "", std::list<std::string>()),
                              std::back_inserter(res3),
                              res1,
                              std::back_inserter(res2));
        CHECK(ret == 3);
        CHECK(res1.empty());
        CHECK(res2.empty());
        CHECK(res3.empty());
      }
    }
  }

  SECTION("Inputs with values") {
    SECTION("Strings") {
      std::string res1, res2;
      auto serial = cs::serialize("foo");
      REQUIRE(serial == "foo"s + cs::detail::kContainerDelimiter);
      auto ret = cs::deserialize(serial, res1);
      CHECK(ret == 3 + 1);
      CHECK(res1 == "foo");

      serial = cs::serialize("foo", "bar");
      REQUIRE(serial == "foo"s + cs::detail::kContainerDelimiter + "bar"s + cs::detail::kContainerDelimiter);
      ret = cs::deserialize(serial, res1, res2);
      CHECK(ret == 3 + 1 + 3 + 1);
      CHECK(res1 == "foo");
      CHECK(res2 == "bar");
    }

    SECTION("Vector of strings") {
      std::vector<std::string> res;
      auto serial = cs::serialize(std::vector<std::string>{"foo"});
      REQUIRE(serial == "foo"s + cs::detail::kElementDelimiter + cs::detail::kContainerDelimiter);
      auto ret = cs::deserialize(serial, std::back_inserter(res));
      CHECK(ret == 3 + 2);
      REQUIRE(res.size() == 1);
      REQUIRE(res[0] == "foo");
      res.clear();

      serial = cs::serialize(std::vector<std::string>{"foo", "bar"});
      REQUIRE(serial == "foo"s + cs::detail::kElementDelimiter + "bar"s + cs::detail::kElementDelimiter +
                            cs::detail::kContainerDelimiter);
      ret = cs::deserialize(serial, std::back_inserter(res));
      CHECK(ret == 3 + 1 + 3 + 2);
      REQUIRE(res.size() == 2);
      CHECK(res[0] == "foo");
      CHECK(res[1] == "bar");
      res.clear();

      serial = cs::serialize(std::vector<std::string>{"foo", "bar", "xyzzy"});
      ret = cs::deserialize(serial, std::back_inserter(res));
      CHECK(ret == serial.size());
      REQUIRE(res.size() == 3);
      CHECK(res[0] == "foo");
      CHECK(res[1] == "bar");
      CHECK(res[2] == "xyzzy");
      res.clear();

      SECTION("Deserialize to list") {
        std::list<std::string> res2;
        ret = cs::deserialize(serial, std::front_inserter(res2));
        CHECK(ret == serial.size());
        REQUIRE(res2.size() == 3);
        auto it = res2.begin();
        CHECK(*it == "xyzzy");
        ++it;
        CHECK(*it == "bar");
        ++it;
        CHECK(*it == "foo");
      }
    }

    SECTION("Vectors of strings") {
      std::vector<std::string> res1, res2;
      ;
      auto serial =
          cs::serialize(std::vector<std::string>{"foo", "bar", "xyzzy"}, std::vector<std::string>{"fred", "wilma"});
      auto ret = cs::deserialize(serial, std::back_inserter(res1), std::back_inserter(res2));
      CHECK(ret == serial.size());
      REQUIRE(res1.size() == 3);
      CHECK(res1[0] == "foo");
      CHECK(res1[1] == "bar");
      CHECK(res1[2] == "xyzzy");
      REQUIRE(res2.size() == 2);
      CHECK(res2[0] == "fred");
      CHECK(res2[1] == "wilma");
    }

    SECTION("Mixture") {
      auto serial = cs::serialize(
          "foobar", std::vector<std::string>{"fred", "wilma"}, "xyzzy", std::list<std::string>{"one", "two", "th ree"});
      std::string res1, res3;
      std::vector<std::string> res2, res4;
      auto ret = cs::deserialize(serial, res1, std::back_inserter(res2), res3, std::back_inserter(res4));
      CHECK(ret == serial.size());
      CHECK(res1 == "foobar");
      REQUIRE(res2.size() == 2);
      CHECK(res2[0] == "fred");
      CHECK(res2[1] == "wilma");
      CHECK(res3 == "xyzzy");
      REQUIRE(res4.size() == 3);
      CHECK(res4[0] == "one");
      CHECK(res4[1] == "two");
      CHECK(res4[2] == "th ree");
    }
  }
  SECTION("Deserialize only part of the serialized content") {
    SECTION("String") {
      std::string res;
      auto serial = cs::serialize("foo", "bar");
      auto ret = cs::deserialize(serial, res);
      CHECK(ret != 0);
      CHECK(ret != serial.size());
      CHECK(res == "foo");
      res.clear();

      serial = cs::serialize("bar", std::vector<std::string>{"foo"});
      ret = cs::deserialize(serial, res);
      CHECK(ret != 0);
      CHECK(ret != serial.size());
      CHECK(res == "bar");
    }

    SECTION("Vector of strings") {
      std::vector<std::string> res;
      auto serial = cs::serialize(std::vector<std::string>{"foo", "bar"}, std::vector<std::string>{"fred", "wilma"});
      auto ret = cs::deserialize(serial, std::back_inserter(res));
      CHECK(ret != 0);
      CHECK(ret != serial.size());
      REQUIRE(res.size() == 2);
      CHECK(res[0] == "foo");
      CHECK(res[1] == "bar");
      res.clear();

      serial = cs::serialize(std::vector<std::string>{"wilma", "fred"}, "fintstones");
      ret = cs::deserialize(serial, std::back_inserter(res));
      CHECK(ret != 0);
      CHECK(ret != serial.size());
      REQUIRE(res.size() == 2);
      CHECK(res[0] == "wilma");
      CHECK(res[1] == "fred");
    }
  }

  SECTION("Serialization error cases") {
    CHECK_THROWS_AS(cs::serialize(""s + cs::detail::kElementDelimiter), cms::Exception);
    CHECK_THROWS_AS(cs::serialize("foo"s + cs::detail::kElementDelimiter), cms::Exception);
    CHECK_THROWS_AS(cs::serialize(cs::detail::kElementDelimiter + "bar"s), cms::Exception);
    CHECK_THROWS_AS(cs::serialize("foo"s + cs::detail::kElementDelimiter + "bar"s), cms::Exception);
    CHECK_THROWS_AS(cs::serialize(""s + cs::detail::kContainerDelimiter), cms::Exception);
    CHECK_THROWS_AS(cs::serialize("foo"s + cs::detail::kContainerDelimiter), cms::Exception);
    CHECK_THROWS_AS(cs::serialize(cs::detail::kContainerDelimiter + "bar"s), cms::Exception);
    CHECK_THROWS_AS(cs::serialize("foo"s + cs::detail::kContainerDelimiter + "bar"s), cms::Exception);

    std::string str = "foo"s + cs::detail::kContainerDelimiter;
    std::vector<std::string> vstr{str};
    CHECK_THROWS_AS(cs::serialize(str, std::vector<std::string>{"foo"}), cms::Exception);
    CHECK_THROWS_AS(cs::serialize(std::vector<std::string>{"foo"}, str), cms::Exception);
    CHECK_THROWS_AS(cs::serialize(vstr, "foo"), cms::Exception);
    CHECK_THROWS_AS(cs::serialize("foo", vstr), cms::Exception);
  }

  SECTION("Deserialization error cases") {
    SECTION("Invalid input") {
      SECTION("Deserializing to string") {
        std::string res;
        CHECK(cs::deserialize("", res) == 0);
        CHECK(cs::deserialize(" ", res) == 0);
        CHECK(cs::deserialize("foo", res) == 0);
        CHECK(cs::deserialize("foo"s + cs::detail::kElementDelimiter + "bar"s, res) == 0);
        CHECK(cs::deserialize("foo"s + cs::detail::kElementDelimiter + "bar"s + cs::detail::kContainerDelimiter, res) ==
              0);
      }

      SECTION("Deserializing to container") {
        std::vector<std::string> res;
        CHECK(cs::deserialize("", std::back_inserter(res)) == 0);
        CHECK(cs::deserialize(" ", std::back_inserter(res)) == 0);
        CHECK(cs::deserialize("foo", std::back_inserter(res)) == 0);
        CHECK(cs::deserialize("foo"s + cs::detail::kElementDelimiter + "bar"s, std::back_inserter(res)) == 0);
        CHECK(cs::deserialize("foo"s + cs::detail::kContainerDelimiter, std::back_inserter(res)) == 0);
      }
    }

    SECTION("Schema mismatch") {
      // Note: empty container and empty string have the same
      // presentation, but this behavior is not tested here as one
      // should not rely on it

      SECTION("Deserializing container as string") {
        std::string res;
        auto ret = cs::deserialize(cs::serialize(std::vector<std::string>{""}), res);
        CHECK(ret == 0);
        ret = cs::deserialize(cs::serialize(std::vector<std::string>{"foo"}), res);
        CHECK(ret == 0);
        ret = cs::deserialize(cs::serialize(std::vector<std::string>{"foo", "bar"}), res);
        CHECK(ret == 0);
      }

      SECTION("Deserializing string as container") {
        std::vector<std::string> res;
        auto ret = cs::deserialize(cs::serialize("foo"), std::back_inserter(res));
        CHECK(ret == 0);
        ret = cs::deserialize(cs::serialize("foo", "bar"), std::back_inserter(res));
        CHECK(ret == 0);
      }
    }

    SECTION("Deserializing too much") {
      SECTION("Strings") {
        std::string res1, res2;
        auto ret = cs::deserialize(cs::serialize("foo"), res1, res2);
        CHECK(ret == 0);
        CHECK(res2.empty());
      }

      SECTION("Vector of strings") {
        std::vector<std::string> res1, res2;
        auto ret = cs::deserialize(
            cs::serialize(std::vector<std::string>{"foo", "bar"}), std::back_inserter(res1), std::back_inserter(res2));
        CHECK(ret == 0);
        CHECK(res2.empty());
      }

      SECTION("Mixture") {
        std::string ress;
        std::vector<std::string> resv;
        auto ret = cs::deserialize(cs::serialize("foo"), ress, std::back_inserter(resv));
        CHECK(ret == 0);
        CHECK(resv.empty());
        ress.clear();

        ret = cs::deserialize(cs::serialize(std::vector<std::string>{"foo"}), std::back_inserter(resv), ress);
        CHECK(ret == 0);
        CHECK(ress.empty());
      }
    }
  }
}
