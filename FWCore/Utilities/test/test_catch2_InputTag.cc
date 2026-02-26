#include "catch2/catch_all.hpp"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace edm {
  class ProductRegistry;

  class TestEDGetToken {
  public:
    static EDGetToken makeEDGetToken(unsigned int iValue) { return EDGetToken(iValue); }
  };
}  // namespace edm

namespace {
  class TypeToTestInputTag1 {};
  class TypeToTestInputTag2 {};
}  // namespace

TEST_CASE("test edm::InputTag", "[InputTag]") {
  constexpr TypeToTestInputTag1 typeToTestInputTag1;
  constexpr TypeToTestInputTag2 typeToTestInputTag2;

  const edm::TypeID testTypeID1(typeToTestInputTag1);
  const edm::TypeID testTypeID2(typeToTestInputTag2);

  SECTION("default constructor") {
    const edm::InputTag tag1;

    REQUIRE(tag1.label() == "");
    REQUIRE(tag1.instance() == "");
    REQUIRE(tag1.process() == "");
    REQUIRE(not tag1.willSkipCurrentProcess());
  }

  SECTION("InputTag(string,string,string)") {
    const edm::InputTag tag2(std::string("a"), std::string("b"), std::string("c"));
    REQUIRE(tag2.label() == "a");
    REQUIRE(tag2.instance() == "b");
    REQUIRE(tag2.process() == "c");
    REQUIRE(not tag2.willSkipCurrentProcess());
  }

  SECTION("InputTag(char*,char*,char*)") {
    const edm::InputTag tag3("d", "e", "f");
    REQUIRE(tag3.label() == "d");
    REQUIRE(tag3.instance() == "e");
    REQUIRE(tag3.process() == "f");
    REQUIRE(not tag3.willSkipCurrentProcess());
  }

  SECTION("InputTag(string) 3 parts") {
    const edm::InputTag tag4("g:h:i");
    REQUIRE(tag4.label() == "g");
    REQUIRE(tag4.instance() == "h");
    REQUIRE(tag4.process() == "i");
    REQUIRE(not tag4.willSkipCurrentProcess());
  }

  SECTION("InputTag(string) 2 parts") {
    const edm::InputTag tag5("g:h");
    REQUIRE(tag5.label() == "g");
    REQUIRE(tag5.instance() == "h");
    REQUIRE(tag5.process() == "");
    REQUIRE(not tag5.willSkipCurrentProcess());
  }

  SECTION("InputTag(string) 1 part") {
    const edm::InputTag tag6("g");
    REQUIRE(tag6.label() == "g");
    REQUIRE(tag6.instance() == "");
    REQUIRE(tag6.process() == "");
    REQUIRE(not tag6.willSkipCurrentProcess());
  }

  SECTION("operator==") {
    const edm::InputTag tag2(std::string("a"), std::string("b"), std::string("c"));
    const edm::InputTag tag4("g:h:i");
    const edm::InputTag tag5("g:h");
    const edm::InputTag tag6("g");
    const edm::InputTag tag7(std::string("a"), std::string("b"), std::string("c"));
    const edm::InputTag tag8(std::string("x"), std::string("b"), std::string("c"));
    REQUIRE(tag2 == tag7);
    REQUIRE(!(tag4 == tag5));
    REQUIRE(!(tag5 == tag6));
    REQUIRE(!(tag7 == tag8));
  }

  SECTION("encode") {
    const edm::InputTag tag5("g:h");
    const edm::InputTag tag6("g");
    const edm::InputTag tag7(std::string("a"), std::string("b"), std::string("c"));

    REQUIRE(tag7.encode() == std::string("a:b:c"));
    REQUIRE(tag5.encode() == std::string("g:h"));
    REQUIRE(tag6.encode() == std::string("g"));
  }

  SECTION("copy ctr") {
    const edm::InputTag tag8(std::string("x"), std::string("b"), std::string("c"));
    const edm::InputTag tag9(tag8);
    REQUIRE(tag8 == tag9);
  }
  SECTION("move ctr") {
    const edm::InputTag tag7(std::string("a"), std::string("b"), std::string("c"));
    edm::InputTag tag11("a:b:c");
    const edm::InputTag tag10(std::move(tag11));
    REQUIRE(tag10 == tag7);
  }
  SECTION("operator=") {
    edm::InputTag tag6("g");
    const edm::InputTag tag10("a:b:c");
    tag6 = tag10;
    REQUIRE(tag10 == tag6);
  }
  SECTION("move operator=") {
    const edm::InputTag tag10("a:b:c");
    edm::InputTag tag5("g:h");
    tag5 = edm::InputTag("a:b:c");
    REQUIRE(tag5 == tag10);
  }

  SECTION("skipCurrentProcess") {
    SECTION("construct with char*") {
      const edm::InputTag tag12("d", "e", "@skipCurrentProcess");
      REQUIRE(tag12.label() == "d");
      REQUIRE(tag12.instance() == "e");
      REQUIRE(tag12.process() == "@skipCurrentProcess");
      REQUIRE(tag12.willSkipCurrentProcess());
    }
    SECTION("construct with std::string") {
      const edm::InputTag tag12a(std::string("d"), std::string("e"), std::string("@skipCurrentProcess"));
      REQUIRE(tag12a.label() == "d");
      REQUIRE(tag12a.instance() == "e");
      REQUIRE(tag12a.process() == "@skipCurrentProcess");
      REQUIRE(tag12a.willSkipCurrentProcess());
    }
    const edm::InputTag tag12b("d:e:@skipCurrentProcess");
    SECTION("construct with one value") {
      REQUIRE(tag12b.label() == "d");
      REQUIRE(tag12b.instance() == "e");
      REQUIRE(tag12b.process() == "@skipCurrentProcess");
      REQUIRE(tag12b.willSkipCurrentProcess());
    }
    SECTION("copy ctr") {
      const edm::InputTag tag12c(tag12b);
      REQUIRE(tag12c.willSkipCurrentProcess());
    }
    SECTION("operator=") {
      edm::InputTag tag12d;
      const edm::InputTag tag12a(std::string("d"), std::string("e"), std::string("@skipCurrentProcess"));
      tag12d = tag12a;
      REQUIRE(tag12d.willSkipCurrentProcess());
    }
    SECTION("test wrong tag") {
      const edm::InputTag tag13("d", "e", "@skipCurrentProcessx");
      REQUIRE(not tag13.willSkipCurrentProcess());

      const edm::InputTag tag14("d", "e", "@skipCurrentProces");
      REQUIRE(not tag14.willSkipCurrentProcess());

      const edm::InputTag tag15("d", "e", "@skipCurrentProcesx");
      REQUIRE(not tag15.willSkipCurrentProcess());
    }
  }
  SECTION("cachedToken") {
    edm::InputTag tag5("g:h");

    auto token = tag5.cachedToken();
    REQUIRE(token.index() == edm::EDGetToken().index());

    tag5.cacheToken(edm::TestEDGetToken::makeEDGetToken(5));
    REQUIRE(tag5.cachedToken().index() == 5);
    tag5.cacheToken(edm::TestEDGetToken::makeEDGetToken(6));
    REQUIRE(tag5.cachedToken().index() == 6);
  }
}
