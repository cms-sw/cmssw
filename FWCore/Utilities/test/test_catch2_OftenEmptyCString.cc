#include "catch.hpp"
#include "FWCore/Utilities/interface/OftenEmptyCString.h"
#include <cstring>

TEST_CASE("test edm::OftenEmptyCString", "[OftenEmptyCString]") {
  SECTION("Constructors") {
    SECTION("default") {
      edm::OftenEmptyCString s;
      REQUIRE(s.c_str() != nullptr);
      REQUIRE(s.c_str()[0] == '\0');
    }
    SECTION("from const *") {
      SECTION("nullptr") {
        edm::OftenEmptyCString s(nullptr);
        REQUIRE(s.c_str() != nullptr);
        REQUIRE(s.c_str()[0] == '\0');
      }
      SECTION("empty") {
        const char* kEmpty = "";
        edm::OftenEmptyCString s(kEmpty);
        REQUIRE(s.c_str() != nullptr);
        REQUIRE(s.c_str() != kEmpty);
        REQUIRE(s.c_str()[0] == '\0');
      }
      SECTION("non empty string") {
        const char* kValue = "something";
        edm::OftenEmptyCString s(kValue);
        REQUIRE(s.c_str() != nullptr);
        REQUIRE(s.c_str() != kValue);
        REQUIRE(strncmp(kValue, s.c_str(), 9) == 0);
        REQUIRE(strlen(kValue) == strlen(s.c_str()));
      }
    }
    SECTION("Copy") {
      SECTION("from non empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy(s);
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(strcmp(s.c_str(), copy.c_str()) == 0);
      }
      SECTION("from default") {
        edm::OftenEmptyCString s;
        edm::OftenEmptyCString copy(s);
        REQUIRE(s.c_str() == copy.c_str());
        REQUIRE(s.c_str() != nullptr);
        REQUIRE(strlen(s.c_str()) == 0);
      }
    }
    SECTION("Move") {
      SECTION("from non empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy(std::move(s));
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(s.c_str() == nullptr);
        REQUIRE(strcmp("something", copy.c_str()) == 0);
      }
      SECTION("from default") {
        edm::OftenEmptyCString s;
        edm::OftenEmptyCString copy(std::move(s));
        REQUIRE(s.c_str() == nullptr);
        REQUIRE(copy.c_str() == edm::OftenEmptyCString().c_str());
      }
    }
  }
  SECTION("operator=") {
    SECTION("copy version") {
      SECTION("from non empty to non empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy("else");
        copy = s;
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(strcmp("something", copy.c_str()) == 0);
        REQUIRE(strcmp(s.c_str(), copy.c_str()) == 0);
      }
      SECTION("from default to non empty") {
        edm::OftenEmptyCString s;
        edm::OftenEmptyCString copy("original");
        copy = s;
        REQUIRE(strcmp(s.c_str(), copy.c_str()) == 0);
      }
      SECTION("from non empty to empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy;
        copy = s;
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(strcmp("something", copy.c_str()) == 0);
        REQUIRE(strcmp(s.c_str(), copy.c_str()) == 0);
      }
    }
    SECTION("move version") {
      SECTION("from non empty to non empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy("else");
        copy = std::move(s);
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(s.c_str() == nullptr);
        REQUIRE(strcmp("something", copy.c_str()) == 0);
      }
      SECTION("from default to non empty") {
        edm::OftenEmptyCString s;
        edm::OftenEmptyCString copy("original");
        copy = std::move(s);
        REQUIRE(copy.c_str() != nullptr);
        REQUIRE(copy.c_str()[0] == '\0');
        REQUIRE(s.c_str() == nullptr);
      }
      SECTION("from non empty to empty") {
        edm::OftenEmptyCString s("something");
        edm::OftenEmptyCString copy;
        copy = std::move(s);
        REQUIRE(s.c_str() != copy.c_str());
        REQUIRE(s.c_str() == nullptr);
        REQUIRE(strcmp("something", copy.c_str()) == 0);
      }
    }
  }
}