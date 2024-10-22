#include "catch.hpp"

#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Exception.h"

using cms::Digest;
using cms::MD5Result;

namespace {
  void testGivenString(std::string const& s) {
    Digest dig1(s);
    MD5Result r1 = dig1.digest();

    Digest dig2;
    dig2.append(s);
    MD5Result r2 = dig2.digest();
    REQUIRE(r1 == r2);

    // The result should be valid *iff* s is non-empty.
    REQUIRE(r1.isValid() == !s.empty());
    REQUIRE(r1.toString().size() == 32);
    REQUIRE(r1.compactForm().size() == 16);
  }
}  // namespace

TEST_CASE("Test cms::Digest", "[Digest]") {
  SECTION("Identical") {
    Digest dig1;
    dig1.append("hello");
    Digest dig2("hello");

    MD5Result r1 = dig1.digest();
    MD5Result r2 = dig2.digest();

    REQUIRE(r1 == r2);
    REQUIRE(!(r1 < r2));
    REQUIRE(!(r2 < r1));

    REQUIRE(r1.toString().size() == 32);
  }
  SECTION("various strings") {
    testGivenString("a");
    testGivenString("{ }");
    testGivenString("abc 123 abc");
  }
  SECTION("empty string") {
    std::string e;
    testGivenString(e);

    Digest dig1;
    MD5Result r1 = dig1.digest();

    MD5Result r2;
    REQUIRE(r1 == r2);

    REQUIRE(!r1.isValid());
  }
  SECTION("conversions") {
    std::string data("aldjfakl\tsdjf34234 \najdf");
    Digest dig(data);
    MD5Result r1 = dig.digest();
    REQUIRE(r1.isValid());
    std::string hexy = r1.toString();
    REQUIRE(hexy.size() == 32);
    MD5Result r2;
    r2.fromHexifiedString(hexy);
    REQUIRE(r1 == r2);
    REQUIRE(r1.toString() == r2.toString());
    REQUIRE(r1.compactForm() == r2.compactForm());

    //check the MD5Result lookup table
    MD5Result lookup;
    MD5Result fromHex;
    for (unsigned int i = 0; i < 256; ++i) {
      for (unsigned int j = 0; j < 16; ++j) {
        lookup.bytes[j] = static_cast<char>(i);
        fromHex.fromHexifiedString(lookup.toString());
        REQUIRE(lookup == fromHex);
        REQUIRE(lookup.toString() == fromHex.toString());
        REQUIRE(lookup.compactForm() == fromHex.compactForm());
      }
    }
  }
}
