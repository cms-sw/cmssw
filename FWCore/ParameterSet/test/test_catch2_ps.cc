/*
 */
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <cassert>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
namespace edm {
  bool operator!=(const edm::EventRange& iLHS, const edm::EventRange& iRHS) {
    return ((iLHS.startEventID() != iRHS.startEventID()) or (iLHS.endEventID() != iRHS.endEventID()));
  }
}  // namespace edm
namespace {
  void do_compare(const edm::EventRange& iLHS, const edm::EventRange& iRHS) {
    CHECK(iLHS.startEventID() == iRHS.startEventID());
    CHECK(iLHS.endEventID() == iRHS.endEventID());
  }

  template <class T>
  void do_compare(const T& iLHS, const T& iRHS) {
    CHECK(iLHS == iRHS);
  }

  template <class T, class A>
  void do_compare(const std::vector<T, A>& iLHS, const std::vector<T, A>& iRHS) {
    REQUIRE_THAT(iLHS, Catch::Matchers::Equals(iRHS));
  }

  template <class T>
  void trackedTestbody(const T& value) {
    edm::ParameterSet p1;
    p1.template addParameter<T>("x", value);
    p1.registerIt();
    do_compare(p1.template getParameter<T>("x"), value);
    std::string p1_encoded = p1.toString();
    edm::ParameterSet p2(p1_encoded);
    REQUIRE(p1 == p2);
    do_compare(p2.template getParameter<T>("x"), value);
  }

  template <class T>
  void untrackedTestbody(const T& value) {
    edm::ParameterSet p;
    p.template addUntrackedParameter<T>("x", value);
    do_compare(p.template getUntrackedParameter<T>("x"), value);

    REQUIRE_THROWS_AS(p.template getUntrackedParameter<T>("does not exist"), cms::Exception);
  }

  template <class T>
  void testbody(T value) {
    trackedTestbody<T>(value);
    untrackedTestbody<T>(value);
  }

  template <class T>
  void test_for_name() {
    edm::ParameterSet preal;
    edm::ParameterSet const& ps = preal;
    // Use 'ps' to make sure we're only getting 'const' access;
    // use 'preal' when we need to modify the underlying ParameterSet.

    std::vector<std::string> names = ps.getParameterNames();
    REQUIRE(names.empty());

    T value{};
    preal.template addParameter<T>("x", value);
    names = ps.getParameterNames();
    REQUIRE(names.size() == 1);
    REQUIRE(names[0] == "x");
    T stored_value = ps.template getParameter<T>(names[0]);
    REQUIRE(stored_value == value);

    preal.template addUntrackedParameter<T>("y", value);
    preal.registerIt();
    names = ps.getParameterNames();
    REQUIRE(names.size() == 2);

    edm::sort_all(names);
    REQUIRE(edm::binary_search_all(names, "x"));
    REQUIRE(edm::binary_search_all(names, "y"));
    names = ps.template getParameterNamesForType<T>();
    REQUIRE(names.size() == 1);
    edm::sort_all(names);
    REQUIRE(edm::binary_search_all(names, "x"));
    names = ps.template getParameterNamesForType<T>(false);
    REQUIRE(names.size() == 1);
    edm::sort_all(names);
    REQUIRE(edm::binary_search_all(names, "y"));

    std::string firstString = ps.toString();
    edm::ParameterSet p2(firstString);

    p2.registerIt();
    // equality tests toStringOfTracked internally
    REQUIRE(ps == p2);
    std::string allString;
    ps.allToString(allString);
    //since have untracked parameter these strings will not be identical
    REQUIRE(firstString != allString);

    edm::ParameterSet pAll(allString);
    REQUIRE(pAll.getParameterNames().size() == 2);
  }

}  // namespace

TEST_CASE("test ParameterSet", "[ParameterSet]") {
  SECTION("empty") {
    edm::ParameterSet p1;
    std::string p1_encoded = p1.toString();
    edm::ParameterSet p2(p1_encoded);
    REQUIRE(p1 == p2);
  }

  SECTION("bool") {
    testbody<bool>(false);
    testbody<bool>(true);
  }

  SECTION("int") {
    testbody<int>(-std::numeric_limits<int>::max());
    testbody<int>(-2112);
    testbody<int>(-0);
    testbody<int>(0);
    testbody<int>(35621);
    testbody<int>(std::numeric_limits<int>::max());
  }

  SECTION("uint") {
    testbody<unsigned int>(0);
    testbody<unsigned int>(35621);
    testbody<unsigned int>(std::numeric_limits<unsigned int>::max());

    testbody<std::vector<unsigned int>>(std::vector<unsigned int>());
    testbody<std::vector<unsigned int>>(std::vector<unsigned int>(1, 35621));
    testbody<std::vector<unsigned int>>(std::vector<unsigned int>(1, std::numeric_limits<unsigned int>::max()));
  }

  SECTION("double") {
    testbody<double>(-1.25);
    testbody<double>(-0.0);
    testbody<double>(0.0);
    testbody<double>(1.25);
    //testbody<double>(1.0/0.0);  // parameter set does not handle infinity?
    //testbody<double>(0.0/0.0);  // parameter set does not handle NaN?
    testbody<double>(-2.3456789e-231);
    testbody<double>(-2.3456789e231);
    testbody<double>(2.3456789e-231);
    testbody<double>(2.3456789e231);
    double oneThird = 1.0 / 3.0;
    testbody<double>(oneThird);
  }

  SECTION("edm::decode_vstring_extent") {
    SECTION("empty") {
      auto ret = edm::decode_vstring_extent("{}");
      REQUIRE(ret);
      CHECK(*ret == "{}");
    }
    SECTION("empty stuff after") {
      auto ret = edm::decode_vstring_extent("{}xava");
      REQUIRE(ret);
      CHECK(*ret == "{}");
    }
    SECTION("one empty string") {
      auto ret = edm::decode_vstring_extent(std::string_view("{\0\0}", 4));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("{\0\0}", 4));
    }
    SECTION("one string with a null") {
      auto ret = edm::decode_vstring_extent(std::string_view("{\0\0\0\0}", 6));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("{\0\0\0\0}", 6));
    }
    SECTION("one empty string, stuff after") {
      auto ret = edm::decode_vstring_extent(std::string_view("{\0\0})aa", 6));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("{\0\0}", 4));
    }
    SECTION("simple multi-entry") {
      auto ret = edm::decode_vstring_extent(std::string_view("{\0one\0,1\0,TWO\0,three\0}", 22));
      REQUIRE(ret);
      CHECK(*ret == std::string_view(std::string_view("{\0one\0,1\0,TWO\0,three\0}", 22)));
    }
    SECTION("simple multi-entry, stuff after") {
      auto ret = edm::decode_vstring_extent(std::string_view("{\0one\0,1\0,TWO\0,three\0});", 24));
      REQUIRE(ret);
      CHECK(*ret == std::string_view(std::string_view("{\0one\0,1\0,TWO\0,three\0}", 22)));
    }
  }
  SECTION("edm::decode_string_extent") {
    SECTION("empty") {
      auto ret = edm::decode_string_extent(std::string_view("\0", 1));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0", 1));
    }
    SECTION("empty stuff after") {
      auto ret = edm::decode_string_extent(std::string_view("\0xava", 5));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0", 1));
    }
    SECTION("string with null") {
      auto ret = edm::decode_string_extent(std::string_view("\0\0\0", 3));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0\0\0", 3));
    }
    SECTION("string with null, stuff after") {
      auto ret = edm::decode_string_extent(std::string_view("\0\0\0)aa", 6));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0\0\0", 3));
    }
    SECTION("null separator") {
      auto ret = edm::decode_string_extent(std::string_view("a\0\0b\0", 5));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("a\0\0b\0", 5));
    }
    SECTION("null separator, stuff after") {
      auto ret = edm::decode_string_extent(std::string_view("a\0\0b\0)>", 7));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("a\0\0b\0", 5));
    }
    SECTION("null start") {
      auto ret = edm::decode_string_extent(std::string_view("\0\0ab\0", 5));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0\0ab\0", 5));
    }
    SECTION("null start, stuff after") {
      auto ret = edm::decode_string_extent(std::string_view("\0\0ab\0)>", 7));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("\0\0ab\0", 5));
    }
    SECTION("null end") {
      auto ret = edm::decode_string_extent(std::string_view("ab\0\0\0", 5));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("ab\0\0\0", 5));
    }
    SECTION("null end, stuff after") {
      auto ret = edm::decode_string_extent(std::string_view("ab\0\0\0)>", 7));
      REQUIRE(ret);
      CHECK(*ret == std::string_view("ab\0\0\0", 5));
    }
  }
  SECTION("string") {
    SECTION("simple") {
      testbody<std::string>("");
      testbody<std::string>("Hello there");
      testbody<std::string>("123");
    }
    SECTION("escaped characters") { testbody<std::string>("This\nis\tsilly\n"); }
    SECTION("special to PSet characters") { testbody<std::string>("{This,is`silly}"); }
    SECTION("all characters") {
      std::array<char, 256> allChars;
      for (int i = 0; i < 256; ++i) {
        allChars[i] = static_cast<char>(i);
      }
      std::string const all(allChars.begin(), allChars.end());
      REQUIRE(all.size() == 256);
      testbody<std::string>(all);
    }
    SECTION("lots of nulls") {
      testbody<std::string>(std::string("\0", 1));
      testbody<std::string>(std::string("\0\0", 2));
      testbody<std::string>(std::string("\0\0\0", 3));
      testbody<std::string>(std::string("\0\0\0\0", 4));
    }
    SECTION("null separator") { testbody<std::string>(std::string("a\0b", 3U)); }
    SECTION("null begin") { testbody<std::string>(std::string("\0ab", 3U)); }
    SECTION("null end") { testbody<std::string>(std::string("ab\0", 3U)); }
    SECTION(":") { testbody<std::string>("ab:c"); }
    SECTION("existsAs") {
      std::string s = "some value";
      edm::ParameterSet p1;
      p1.addParameter<std::string>("s", s);
      p1.registerIt();
      REQUIRE(p1.existsAs<std::string>("s"));
      REQUIRE(not p1.existsAs<std::string>("not_here"));
    }
  }
  SECTION("vstring") {
    SECTION("simple") {
      std::vector<std::string> vs;
      vs.push_back("one");
      vs.push_back("1");
      vs.push_back("TWO");
      vs.push_back("three");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("vstring with empty strings") {
      std::vector<std::string> vs;
      vs.push_back("");
      vs.push_back("1");
      vs.push_back("");
      vs.push_back("three");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("empty vstring") {
      std::vector<std::string> vs;
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("with one empty string") {
      std::vector<std::string> vs;
      vs.push_back("");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("special characters") {
      //special characters
      std::vector<std::string> vs;
      vs.push_back("{This,is`silly}");
      vs.push_back("1");
      vs.push_back("three");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading '}'") {
      std::vector<std::string> vs;
      vs.push_back("}");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading ','") {
      std::vector<std::string> vs;
      vs.push_back(",");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading '<'") {
      std::vector<std::string> vs;
      vs.push_back("<");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading '>'") {
      std::vector<std::string> vs;
      vs.push_back(">");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading ';'") {
      std::vector<std::string> vs;
      vs.push_back(";");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("leading ':'") {
      std::vector<std::string> vs;
      vs.push_back(":");
      testbody<std::vector<std::string>>(vs);
    }
    SECTION("existsAs") {
      std::vector<std::string> vs;
      vs.push_back("some value");
      edm::ParameterSet p1;
      p1.addParameter<std::vector<std::string>>("vs", vs);
      p1.registerIt();
      REQUIRE(p1.existsAs<std::vector<std::string>>("vs"));
      REQUIRE(not p1.existsAs<std::vector<std::string>>("not_here"));
    }
  }
  SECTION("deprecated") {
    SECTION("string") {
      std::string to;
      std::string const value = "this is deprecated";
      REQUIRE(edm::encode_deprecated(to, value));
      std::string pset_encode;
      pset_encode += "<label=+S(";
      pset_encode += to;
      pset_encode += ")>";
      edm::ParameterSet pset(pset_encode);
      pset.registerIt();
      REQUIRE(pset.getParameter<std::string>("label") == value);
      REQUIRE(pset.existsAs<std::string>("label"));
    }
    SECTION("vstring") {
      std::string to;
      std::string const value = "this is deprecated";
      std::vector<std::string> const from(1, value);
      REQUIRE(edm::encode_deprecated(to, from));
      std::string pset_encode;
      pset_encode += "<label=+s(";
      pset_encode += to;
      pset_encode += ")>";
      edm::ParameterSet pset(pset_encode);
      pset.registerIt();
      REQUIRE(pset.getParameter<std::vector<std::string>>("label") == from);
      REQUIRE(pset.existsAs<std::vector<std::string>>("label"));
    }
  }
  SECTION("eventID") {
    testbody<edm::EventID>(edm::EventID());
    testbody<edm::EventID>(edm::EventID::firstValidEvent());
    testbody<edm::EventID>(edm::EventID(2, 3, 4));
    testbody<edm::EventID>(edm::EventID(2, 3, edm::EventID::maxEventNumber()));
    testbody<edm::EventID>(edm::EventID(
        edm::EventID::maxRunNumber(), edm::EventID::maxLuminosityBlockNumber(), edm::EventID::maxEventNumber()));
  }

  SECTION("eventRange") {
    testbody<edm::EventRange>(edm::EventRange());
    testbody<edm::EventRange>(edm::EventRange(1,
                                              1,
                                              1,
                                              edm::EventID::maxRunNumber(),
                                              edm::EventID::maxLuminosityBlockNumber(),
                                              edm::EventID::maxEventNumber()));
    testbody<edm::EventRange>(edm::EventRange(2, 3, 4, 2, 3, 10));

    testbody<edm::EventRange>(
        edm::EventRange(1, 0, 1, edm::EventID::maxRunNumber(), 0, edm::EventID::maxEventNumber()));
    testbody<edm::EventRange>(edm::EventRange(2, 0, 4, 2, 0, 10));
  }

  SECTION("vEventRange") {
    testbody<std::vector<edm::EventRange>>(std::vector<edm::EventRange>());
    testbody<std::vector<edm::EventRange>>(
        std::vector<edm::EventRange>(1,
                                     edm::EventRange(1,
                                                     1,
                                                     1,
                                                     edm::EventID::maxRunNumber(),
                                                     edm::EventID::maxLuminosityBlockNumber(),
                                                     edm::EventID::maxEventNumber())));

    testbody<std::vector<edm::EventRange>>(std::vector<edm::EventRange>(
        1, edm::EventRange(1, 0, 1, edm::EventID::maxRunNumber(), 0, edm::EventID::maxEventNumber())));

    std::vector<edm::EventRange> er;
    er.reserve(2);
    er.push_back(edm::EventRange(2, 3, 4, 2, 3, 10));
    er.push_back(edm::EventRange(5, 1, 1, 10, 3, 10));

    testbody<std::vector<edm::EventRange>>(er);
  }

  SECTION("fileInPath") {
    edm::ParameterSet p;
    edm::FileInPath fip("FWCore/ParameterSet/python/Config.py");
    p.addParameter<edm::FileInPath>("fip", fip);
    REQUIRE(p.existsAs<edm::FileInPath>("fip"));
    REQUIRE(p.getParameterNamesForType<edm::FileInPath>()[0] == "fip");
  }

  SECTION("InputTag") {
    SECTION("label only") {
      edm::InputTag t("foo");
      testbody<edm::InputTag>(t);
    }
    SECTION("label and instance") {
      edm::InputTag t("foo", "bar");
      testbody<edm::InputTag>(t);
    }
    SECTION("label, instance and process") {
      edm::InputTag t("foo", "bar", "PROC");
      testbody<edm::InputTag>(t);
    }
    SECTION("from string, label only") {
      edm::ParameterSet p;
      p.addParameter<std::string>("tag", "foo");
      p.registerIt();
      auto t = p.getParameter<edm::InputTag>("tag");
      REQUIRE(t.label() == "foo");
      REQUIRE(t.instance().empty());
      REQUIRE(t.process().empty());
    }
    SECTION("from string, label & instance") {
      edm::ParameterSet p;
      p.addParameter<std::string>("tag", "foo:bar");
      p.registerIt();
      auto t = p.getParameter<edm::InputTag>("tag");
      REQUIRE(t.label() == "foo");
      REQUIRE(t.instance() == "bar");
      REQUIRE(t.process().empty());
    }
    SECTION("from string, label , instance & process") {
      edm::ParameterSet p;
      p.addParameter<std::string>("tag", "foo:bar:PROC");
      p.registerIt();
      auto t = p.getParameter<edm::InputTag>("tag");
      REQUIRE(t.label() == "foo");
      REQUIRE(t.instance() == "bar");
      REQUIRE(t.process() == "PROC");
    }
  }

  SECTION("doubleEquality") {
    edm::ParameterSet p1, p2, p3;
    p1.addParameter<double>("x", 0.1);
    p2.addParameter<double>("x", 1.0e-1);
    p3.addParameter<double>("x", 0.100);
    p1.registerIt();
    p2.registerIt();
    p3.registerIt();
    REQUIRE(p1 == p2);
    REQUIRE(p1 == p3);
    REQUIRE(p2 == p3);

    REQUIRE(p1.toString() == p2.toString());
    REQUIRE(p1.toString() == p3.toString());
    REQUIRE(p2.toString() == p3.toString());
  }

  SECTION("negativeZero") {
    edm::ParameterSet a1, a2;
    a1.addParameter<double>("x", 0.0);
    a2.addParameter<double>("x", -0.0);
    a1.registerIt();
    a2.registerIt();
    // Negative and positive zero should be coded differently.
    REQUIRE(a1.toString() != a2.toString());
    REQUIRE(a1 != a2);
    // Negative and positive zero should test equal.
    REQUIRE(a1.getParameter<double>("x") == a2.getParameter<double>("x"));
  }

  SECTION("id") {
    edm::ParameterSet a;
    a.registerIt();
    edm::ParameterSetID a_id = a.id();
    edm::ParameterSet b;
    b.addParameter<int>("x", -23);
    b.registerIt();
    edm::ParameterSetID b_id = b.id();

    REQUIRE(a != b);
    REQUIRE(a.id() != b.id());

    {
      //Check that changes to ESInputTag do not affect ID
      // as that would affect reading back stored PSets

      edm::ParameterSet ps;
      ps.addParameter<edm::ESInputTag>("default", edm::ESInputTag());
      ps.addParameter<edm::ESInputTag>("moduleOnly", edm::ESInputTag("Prod", ""));
      ps.addParameter<edm::ESInputTag>("dataOnly", edm::ESInputTag("", "data"));
      ps.addParameter<edm::ESInputTag>("allLabels", edm::ESInputTag("Prod", "data"));
      ps.registerIt();

      std::string stringValue;
      ps.id().toString(stringValue);
      REQUIRE(stringValue == "01642a9a7311dea2df2f9ee430855a99");
    }
  }

  SECTION("calculateID") {
    std::vector<int> thousand(1000, 0);
    std::vector<int> hundred(100, 0);
    edm::ParameterSet a;
    edm::ParameterSet b;
    edm::ParameterSet c;
    a.addParameter<double>("pi", 3.14);
    a.addParameter<int>("answer", 42), a.addParameter<std::string>("amiga", "rules");
    a.addParameter<std::vector<int>>("thousands", thousand);
    a.addUntrackedParameter<double>("e", 2.72);
    a.addUntrackedParameter<int>("question", 41 + 1);
    a.addUntrackedParameter<std::string>("atari", "too");
    a.addUntrackedParameter<std::vector<int>>("hundred", hundred);

    b.addParameter<double>("pi", 3.14);
    b.addParameter<int>("answer", 42), b.addParameter<std::string>("amiga", "rules");
    b.addParameter<std::vector<int>>("thousands", thousand);
    b.addUntrackedParameter<double>("e", 2.72);
    b.addUntrackedParameter<int>("question", 41 + 1);
    b.addUntrackedParameter<std::string>("atari", "too");
    b.addUntrackedParameter<std::vector<int>>("hundred", hundred);

    c.addParameter<double>("pi", 3.14);
    c.addParameter<int>("answer", 42), c.addParameter<std::string>("amiga", "rules");
    c.addParameter<std::vector<int>>("thousands", thousand);
    c.addUntrackedParameter<double>("e", 2.72);
    c.addUntrackedParameter<int>("question", 41 + 1);
    c.addUntrackedParameter<std::string>("atari", "too");
    c.addUntrackedParameter<std::vector<int>>("hundred", hundred);

    b.addParameter<edm::ParameterSet>("nested", c);
    std::vector<edm::ParameterSet> vb;
    vb.push_back(b);

    a.addUntrackedParameter<std::vector<edm::ParameterSet>>("vps", vb);
    std::string stringrep;
    a.toString(stringrep);
    cms::Digest md5alg(stringrep);
    cms::Digest newDigest;
    a.toDigest(newDigest);
    REQUIRE(md5alg.digest().toString() == newDigest.digest().toString());
  }

  SECTION("mapById") {
    // makes parameter sets and ids
    edm::ParameterSet a;
    a.addParameter<double>("pi", 3.14);
    a.addParameter<std::string>("name", "Bub");
    a.registerIt();
    REQUIRE(a.exists("pi"));
    REQUIRE(!a.exists("pie"));

    edm::ParameterSet b;
    b.addParameter<bool>("b", false);
    b.addParameter<std::vector<int>>("three_zeros", std::vector<int>(3, 0));
    b.registerIt();

    edm::ParameterSet c;
    c.registerIt();

    edm::ParameterSet d;
    d.addParameter<unsigned int>("hundred", 100);
    d.addParameter<std::vector<double>>("empty", std::vector<double>());
    d.registerIt();

    edm::ParameterSetID id_a = a.id();
    edm::ParameterSetID id_b = b.id();
    edm::ParameterSetID id_c = c.id();
    edm::ParameterSetID id_d = d.id();

    // fill map
    typedef std::map<edm::ParameterSetID, edm::ParameterSet> map_t;
    map_t psets;

    psets.insert(std::make_pair(id_a, a));
    psets.insert(std::make_pair(id_b, b));
    psets.insert(std::make_pair(id_c, c));
    psets.insert(std::make_pair(id_d, d));

    // query map
    REQUIRE(psets.size() == 4);
    REQUIRE(psets[id_a] == a);
    REQUIRE(psets[id_b] == b);
    REQUIRE(psets[id_c] == c);
    REQUIRE(psets[id_d] == d);
  }

  SECTION("name Access") {
    test_for_name<bool>();

    test_for_name<int>();
    test_for_name<std::vector<int>>();

    test_for_name<unsigned int>();
    test_for_name<std::vector<unsigned int>>();

    test_for_name<double>();
    test_for_name<std::vector<double>>();

    test_for_name<std::string>();
    test_for_name<std::vector<std::string>>();
    test_for_name<edm::ParameterSet>();
    test_for_name<std::vector<edm::ParameterSet>>();

    // Can't make default FileInPath objects...

    // Now make sure that if we put in a parameter of type A, we don't
    // see it when we ask for names of type B != A.
    {
      edm::ParameterSet p;
      p.addParameter<double>("a", 2.5);
      p.registerIt();
      const bool tracked = true;
      std::vector<std::string> names = p.getParameterNamesForType<int>(tracked);
      REQUIRE(names.empty());
    }
  }

  SECTION("Embedded PSet") {
    edm::ParameterSet ps;
    edm::ParameterSet psEmbedded, psDeeper;
    psEmbedded.addUntrackedParameter<std::string>("p1", "wham");
    psEmbedded.addParameter<std::string>("p2", "bam");
    psDeeper.addParameter<int>("deepest", 6);
    psDeeper.registerIt();
    edm::InputTag it("label", "instance");
    std::vector<edm::InputTag> vit;
    vit.push_back(it);
    psEmbedded.addParameter<edm::InputTag>("it", it);
    psEmbedded.addParameter<std::vector<edm::InputTag>>("vit", vit);
    psEmbedded.addParameter<edm::ParameterSet>("psDeeper", psDeeper);
    psEmbedded.registerIt();
    ps.addParameter<edm::ParameterSet>("psEmbedded", psEmbedded);
    ps.addParameter<double>("topLevel", 1.);
    ps.addUntrackedParameter<unsigned long long>("u64", 64);

    std::vector<edm::ParameterSet> vpset;
    edm::ParameterSet pset1;
    pset1.addParameter<int>("int1", 1);
    edm::ParameterSet pset2;
    pset2.addParameter<int>("int2", 2);
    edm::ParameterSet pset3;
    pset3.addParameter<int>("int3", 3);
    vpset.push_back(pset1);
    vpset.push_back(pset2);
    vpset.push_back(pset3);
    ps.addParameter<std::vector<edm::ParameterSet>>("psVPset", vpset);

    ps.registerIt();

    std::string rep = ps.toString();
    edm::ParameterSet defrosted(rep);
    defrosted.registerIt();
    edm::ParameterSet trackedPart(ps.trackedPart());

    REQUIRE(defrosted == ps);
    REQUIRE(trackedPart.exists("psEmbedded"));
    REQUIRE(trackedPart.getParameterSet("psEmbedded").exists("p2"));
    REQUIRE(!trackedPart.getParameterSet("psEmbedded").exists("p1"));
    REQUIRE(trackedPart.getParameterSet("psEmbedded").getParameterSet("psDeeper").getParameter<int>("deepest") == 6);
    REQUIRE(ps.getUntrackedParameter<unsigned long long>("u64") == 64);
    REQUIRE(!trackedPart.exists("u64"));
    std::vector<edm::ParameterSet> const& vpset1 = trackedPart.getParameterSetVector("psVPset");
    REQUIRE(vpset1[0].getParameter<int>("int1") == 1);
    REQUIRE(vpset1[1].getParameter<int>("int2") == 2);
    REQUIRE(vpset1[2].getParameter<int>("int3") == 3);

    SECTION("deprecated pset encoding") {
      //Used in Utilities/StorageFactory
      char const* const psetChar =
          "<destinations=-s({63657272})"                                  // cerr
          ";cerr=-P(<noTimeStamps=-B(true);threshold=-S(5741524e494e47)"  // WARNING
          ";WARNING=-P(<limit=-I(+0)>);default=-P(<limit=-I(-1)>)>)>";
      edm::ParameterSet pset(psetChar);
      pset.registerIt();
      {
        auto p = pset.getUntrackedParameter<std::vector<std::string>>("destinations");
        REQUIRE(p.size() == 1);
        REQUIRE(p[0] == "cerr");
        REQUIRE(not pset.exists("threshold"));
      }
      {
        auto p = pset.getUntrackedParameter<edm::ParameterSet>("cerr");
        REQUIRE(p.getUntrackedParameter<bool>("noTimeStamps"));
        REQUIRE(p.getUntrackedParameter<std::string>("threshold") == "WARNING");
        {
          auto p2 = p.getUntrackedParameter<edm::ParameterSet>("WARNING");
          REQUIRE(p2.getUntrackedParameter<int>("limit") == 0);
          REQUIRE(not p2.exists("default"));
        }
        {
          auto p3 = p.getUntrackedParameter<edm::ParameterSet>("default");
          REQUIRE(p3.getUntrackedParameter<int>("limit") == -1);
        }
      }
    }
    SECTION("deprecated vstring encoding") {
      char const* const psetChar =
          "<empty=-p({});"
          "simple=-p({<a=-I(+0)>,<a=-I(+1)>,<b=-i({+1,+2})>});"
          "recurse=-p({<r=-p({<a=-I(+0)>,<a=-I(+1)>})>})"
          ">";
      edm::ParameterSet pset(psetChar);
      pset.registerIt();
      {
        auto empty = pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("empty");
        REQUIRE(empty.empty());
      }
      {
        auto simple = pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("simple");
        REQUIRE(simple.size() == 3);
        REQUIRE(simple[0].getUntrackedParameter<int>("a") == 0);
        REQUIRE(simple[1].getUntrackedParameter<int>("a") == 1);
        REQUIRE(simple[2].getUntrackedParameter<std::vector<int>>("b").size() == 2);
      }
      {
        auto recurse = pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("recurse");
        REQUIRE(recurse.size() == 1);
        REQUIRE(recurse[0].getUntrackedParameter<std::vector<edm::ParameterSet>>("r").size() == 2);
      }
    }
  }

  SECTION("Registration") {
    edm::ParameterSet ps;
    edm::ParameterSet psEmbedded, psDeeper;
    psEmbedded.addUntrackedParameter<std::string>("p1", "wham");
    psEmbedded.addParameter<std::string>("p2", "bam");
    psDeeper.addParameter<int>("deepest", 6);
    psDeeper.registerIt();
    edm::InputTag it("label", "instance");
    std::vector<edm::InputTag> vit;
    vit.push_back(it);
    psEmbedded.addParameter<edm::InputTag>("it", it);
    psEmbedded.addParameter<std::vector<edm::InputTag>>("vit", vit);
    psEmbedded.addParameter<edm::ParameterSet>("psDeeper", psDeeper);
    psEmbedded.registerIt();
    ps.addParameter<edm::ParameterSet>("psEmbedded", psEmbedded);
    ps.addParameter<double>("topLevel", 1.);
    ps.addUntrackedParameter<unsigned long long>("u64", 64);
    ps.registerIt();
    REQUIRE(ps.isRegistered());
    REQUIRE(psEmbedded.isRegistered());
    REQUIRE(psDeeper.isRegistered());
    psEmbedded.addParameter<std::string>("p3", "slam");
    REQUIRE(ps.isRegistered());
    REQUIRE(!psEmbedded.isRegistered());
    REQUIRE(psDeeper.isRegistered());
  }

  SECTION("Copy From") {
    edm::ParameterSet psOld;
    edm::ParameterSet psNew;
    edm::ParameterSet psInternal;
    std::vector<edm::ParameterSet> vpset;
    vpset.push_back(psInternal);
    psOld.addParameter<int>("i", 5);
    psOld.addParameter<edm::ParameterSet>("ps", psInternal);
    psOld.addParameter<std::vector<edm::ParameterSet>>("vps", vpset);
    psNew.copyFrom(psOld, "i");
    psNew.copyFrom(psOld, "ps");
    psNew.copyFrom(psOld, "vps");
    REQUIRE(psNew.existsAs<int>("i"));
    REQUIRE(psNew.existsAs<edm::ParameterSet>("ps"));
    REQUIRE(psNew.existsAs<std::vector<edm::ParameterSet>>("vps"));
  }

  SECTION("Get Parameter As String") {
    edm::ParameterSet ps;
    edm::ParameterSet psInternal;
    std::vector<edm::ParameterSet> vpset;
    vpset.push_back(psInternal);
    ps.addParameter<int>("i", 5);
    ps.addParameter<edm::ParameterSet>("ps", psInternal);
    ps.addParameter<std::vector<edm::ParameterSet>>("vps", vpset);
    ps.registerIt();
    psInternal.registerIt();
    std::string parStr = ps.getParameterAsString("i");
    std::string psetStr = ps.getParameterAsString("ps");
    std::string vpsetStr = ps.getParameterAsString("vps");
    std::string parStr2 = ps.retrieve("i").toString();
    std::string psetStr2 = ps.retrieveParameterSet("ps").toString();
    std::string vpsetStr2 = ps.retrieveVParameterSet("vps").toString();
    REQUIRE(parStr == parStr2);
    REQUIRE(psetStr == psetStr2);
    REQUIRE(vpsetStr == vpsetStr2);
  }
}
