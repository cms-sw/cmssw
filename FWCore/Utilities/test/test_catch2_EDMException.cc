#include "catch.hpp"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>
#include <memory>

namespace {
  void func3() {
    edm::Exception ex(edm::errors::NotFound);
    ex << "This is just a test";
    ex.addContext("new1");
    ex.addAdditionalInfo("info1");
    throw ex;
  }

  void func2() { func3(); }

  void func1() {
    try {
      func2();
    } catch (edm::Exception& e) {
      edm::Exception toThrow(edm::errors::Unknown, "In func2", e);
      edm::Exception toThrowString(edm::errors::Unknown, std::string("In func2"), e);
      REQUIRE(toThrow.explainSelf() == toThrowString.explainSelf());
      toThrow << "\nGave up";
      toThrow.addContext("new2");
      toThrow.addAdditionalInfo("info2");
      REQUIRE(toThrow.returnCode() == 8003);
      REQUIRE(toThrow.categoryCode() == edm::errors::Unknown);
      cms::Exception* ptr = &toThrow;
      ptr->raise();
    }
  }

  const char answer[] =
      "An exception of category 'Unknown' occurred while\n"
      "   [0] new2\n"
      "   [1] new1\n"
      "Exception Message:\n"
      "In func2\n"
      "This is just a test\n"
      "Gave up\n"
      "   Additional Info:\n"
      "      [a] info2\n"
      "      [b] info1\n";
}  // namespace

TEST_CASE("Test edm::Exception", "[edm::Exception]") {
  SECTION("throw") { REQUIRE_THROWS_WITH(func1(), answer); }
  edm::Exception ex(edm::errors::NotFound);
  ex << "This is just a test";
  ex.addContext("new1");
  ex.addAdditionalInfo("info1");

  SECTION("copy constructor") {
    edm::Exception cpy(ex);
    REQUIRE(ex.explainSelf() == cpy.explainSelf());
  }
  edm::Exception e1(edm::errors::Unknown, "blah");
  SECTION("returnCode") {
    REQUIRE(e1.returnCode() == 8003);
    REQUIRE(ex.returnCode() == 8026);
  }
  SECTION("category") {
    REQUIRE(e1.category() == std::string("Unknown"));
    REQUIRE(ex.category() == std::string("NotFound"));
  }
  SECTION("message") { REQUIRE(e1.message() == std::string("blah ")); }

  SECTION("equivalence") {
    edm::Exception e1String(edm::errors::Unknown, std::string("blah"));
    REQUIRE(e1.explainSelf() == e1String.explainSelf());
  }
  SECTION("clone") {
    cms::Exception* ptr = &e1;
    cms::Exception* ptrCloneCopy = ptr->clone();
    REQUIRE(ptrCloneCopy->returnCode() == 8003);
    REQUIRE(e1.explainSelf() == ptrCloneCopy->explainSelf());
  }
  SECTION("throwThis") {
    REQUIRE_THROWS_WITH(edm::Exception::throwThis(edm::errors::ProductNotFound, "a", "b", "c", "d", "e"),
                        "An exception of category 'ProductNotFound' occurred.\n"
                        "Exception Message:\n"
                        "a bcde\n");
    REQUIRE_THROWS_WITH(edm::Exception::throwThis(edm::errors::ProductNotFound, "a", 1, "b"),
                        "An exception of category 'ProductNotFound' occurred.\n"
                        "Exception Message:\n"
                        "a 1b\n");
  }
}
