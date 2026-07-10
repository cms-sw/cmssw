/*
 *  parametersetid_t.cppunit.cc
 *  CMSSW
 *
 */

#include <map>
#include <string>

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace {
  std::string default_id_string;
  std::string cow16;

  void setUp() {
    default_id_string = "d41d8cd98f00b204e9800998ecf8427e";
    cow16 = "DEADBEEFDEADBEEF";
  }
}  // namespace

TEST_CASE("ParameterSetID", "[ParameterSetID]") {
  setUp();

  SECTION("constructTest") {
    edm::ParameterSetID id1;
    REQUIRE(!id1.isValid());

    edm::ParameterSetID id2(cow16);
    REQUIRE(id2.isValid());
    REQUIRE(id2.compactForm() == cow16);

    // Test badConstructTest - should throw
    REQUIRE_THROWS_AS(edm::ParameterSetID("1"), edm::Exception);
  }

  SECTION("comparisonTest") {
    edm::ParameterSetID a;
    edm::ParameterSetID b;
    REQUIRE(a == b);
    REQUIRE(!(a != b));
    REQUIRE(!(a < b));
    REQUIRE(!(a > b));
  }

  SECTION("suitableForMapTest") {
    typedef std::map<edm::ParameterSetID, int> map_t;
    map_t m;
    REQUIRE(m.empty());

    edm::ParameterSetID a;
    m[a] = 100;
    REQUIRE(m.size() == 1);
    REQUIRE(m[a] == 100);

    edm::ParameterSetID b(cow16);
    m[b] = 200;
    REQUIRE(m.size() == 2);
    REQUIRE(m[a] == 100);
    REQUIRE(m[b] == 200);

    REQUIRE(m.erase(a) == 1);
    REQUIRE(m.size() == 1);
    REQUIRE(m[b] == 200);
    REQUIRE(m.find(a) == m.end());
  }

  SECTION("unhexifyTest") {
    // 'a' has the MD5 checksum for an empty string.
    edm::ParameterSetID a(default_id_string);
    std::string a_compact = a.compactForm();
    REQUIRE(static_cast<unsigned char>(a_compact[0]) == 0xd4);
    REQUIRE(static_cast<unsigned char>(a_compact[1]) == 0x1d);
    REQUIRE(static_cast<unsigned char>(a_compact[2]) == 0x8c);
    REQUIRE(static_cast<unsigned char>(a_compact[3]) == 0xd9);
    REQUIRE(static_cast<unsigned char>(a_compact[4]) == 0x8f);
    REQUIRE(static_cast<unsigned char>(a_compact[5]) == 0x00);
    REQUIRE(static_cast<unsigned char>(a_compact[6]) == 0xb2);
    REQUIRE(static_cast<unsigned char>(a_compact[7]) == 0x04);
    REQUIRE(static_cast<unsigned char>(a_compact[8]) == 0xe9);
    REQUIRE(static_cast<unsigned char>(a_compact[9]) == 0x80);
    REQUIRE(static_cast<unsigned char>(a_compact[10]) == 0x09);
    REQUIRE(static_cast<unsigned char>(a_compact[11]) == 0x98);
    REQUIRE(static_cast<unsigned char>(a_compact[12]) == 0xec);
    REQUIRE(static_cast<unsigned char>(a_compact[13]) == 0xf8);
    REQUIRE(static_cast<unsigned char>(a_compact[14]) == 0x42);
    REQUIRE(static_cast<unsigned char>(a_compact[15]) == 0x7e);

    edm::ParameterSetID b;
    std::string b_compact = b.compactForm();
    REQUIRE(b_compact.size() == 16);
  }

  SECTION("printTest") {
    std::ostringstream os;
    edm::ParameterSetID id(default_id_string);
    os << id;
    std::string output = os.str();
    REQUIRE(output == default_id_string);

    std::ostringstream os2;
    std::string s2("0123456789abcdef0123456789abcdef");
    edm::ParameterSetID id2(s2);
    REQUIRE(id2.isValid());
    os2 << id2;
    std::string output2 = os2.str();
    REQUIRE(output2 == s2);
  }

  SECTION("oldRootFileCompatibilityTest") {
    using namespace edm;
    //simulate what ROOT does when reading an old ParameterSetID which has 32 characters
    ParameterSetID dflt(default_id_string);
    std::string sValue(default_id_string);
    ParameterSetID* evil(reinterpret_cast<ParameterSetID*>(&sValue));

    REQUIRE(not evil->isCompactForm());
    REQUIRE(dflt.isCompactForm());

    ParameterSetID evilCopy(*evil);
    REQUIRE(evilCopy.isCompactForm());

    REQUIRE(dflt == evilCopy);
    REQUIRE(evilCopy == *evil);

    REQUIRE(dflt == *evil);

    /*Do an 'exhaustive' test to see if comparisons are preserved
    in the case of conversion from non-compact to compact form
    and that comparision between non-compact to compact form also 
    preserves ordering.
    Because the 'hex' version is just a repetition of two characters per byte,
    we only need to do 2^8-1 comparisions rather than 2^32-1 comparisions when
    doing the exhaustive test
   */

    const char hexbits[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    const size_t nHexBits = sizeof(hexbits) / sizeof(char);
    char buffer[3];
    buffer[2] = 0;
    std::string theOldValue("00000000000000000000000000000000");
    ParameterSetID theOldHash(theOldValue);
    for (const char* itHigh = hexbits; itHigh != hexbits + nHexBits; ++itHigh) {
      const char* lowStart = hexbits;
      if (itHigh == hexbits) {
        lowStart += 1;
      }
      for (const char* itLow = lowStart; itLow != hexbits + nHexBits; ++itLow) {
        buffer[0] = *itHigh;
        buffer[1] = *itLow;
        std::string theValue(buffer);
        //need to make this 32 bytes long, now we are 2 bytes
        theValue = theValue + theValue;  //4
        theValue = theValue + theValue;  //8
        theValue = theValue + theValue;  //16
        theValue = theValue + theValue;  //32
        //std::cout <<theValue<<std::endl;
        REQUIRE(theOldValue < theValue);
        ParameterSetID theHash(theValue);
        REQUIRE(theOldHash < theHash);

        ParameterSetID* theEvil(reinterpret_cast<ParameterSetID*>(&theValue));
        ParameterSetID* theOldEvil(reinterpret_cast<ParameterSetID*>(&theOldValue));
        REQUIRE(*theOldEvil < *theEvil);
        REQUIRE(*theOldEvil < theHash);
        REQUIRE(theOldHash < *theEvil);
        theOldValue = theValue;
        theOldHash = theHash;
      }
    }
  }
}
