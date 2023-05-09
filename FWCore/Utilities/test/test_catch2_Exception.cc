#include "catch.hpp"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {
  struct Thing {
    Thing() : x() {}
    explicit Thing(int xx) : x(xx) {}
    int x;
  };

  std::ostream& operator<<(std::ostream& os, const Thing& t) {
    os << "Thing(" << t.x << ")";
    return os;
  }

  constexpr char expected[] =
      "An exception of category 'InfiniteLoop' occurred.\n"
      "Exception Message:\n"
      "In func1\n"
      "This is just a test: \n"
      "double: 1.11111\n"
      "float:  2.22222\n"
      "uint:   75\n"
      "string: a string\n"
      "char*:  a nonconst pointer\n"
      "char[]: a c-style array\n"
      "Thing:  Thing(4)\n"
      "\n"
      "double: 1.111110e+00\n"
      "float:  2.22e+00\n"
      "char*:  ..a nonconst pointer\n"
      "\n"
      "Gave up\n";

  void func3() {
    double d = 1.11111;
    float f = 2.22222;
    unsigned int i = 75U;
    std::string s("a string");
    char* c1 = const_cast<char*>("a nonconst pointer");
    char c2[] = "a c-style array";
    Thing thing(4);

    //  throw cms::Exception("DataCorrupt")
    cms::Exception e("DataCorrupt");
    e << "This is just a test: \n"
      << "double: " << d << "\n"
      << "float:  " << f << "\n"
      << "uint:   " << i << "\n"
      << "string: " << s << "\n"
      << "char*:  " << c1 << "\n"
      << "char[]: " << c2 << "\n"
      << "Thing:  " << thing << "\n"
      << std::endl
      << "double: " << std::scientific << d << "\n"
      << "float:  " << std::setprecision(2) << f << "\n"
      << "char*:  " << std::setfill('.') << std::setw(20) << c1 << std::setfill(' ') << "\n"
      << std::endl;

    throw e;
  }

  void func2() { func3(); }

  void func1() {
    try {
      func2();
    } catch (cms::Exception& e) {
      cms::Exception toThrow("InfiniteLoop", "In func1", e);
      toThrow << "Gave up";
      throw toThrow;
    }
  }

}  // namespace

TEST_CASE("Test cms::Exception", "[cms::Exception]") {
  SECTION("throw") { REQUIRE_THROWS_WITH(func1(), expected); }
  SECTION("returnCode") {
    cms::Exception e1("ABC");
    REQUIRE(e1.returnCode() == 8001);
  }
  SECTION("alreadyPrinted") {
    cms::Exception e1("ABC");
    REQUIRE(not e1.alreadyPrinted());
    e1.setAlreadyPrinted();
    REQUIRE(e1.alreadyPrinted());
    SECTION("copy constructor") {
      cms::Exception e("ABC");
      cms::Exception e2(e);
      REQUIRE(not e2.alreadyPrinted());

      e.setAlreadyPrinted();
      cms::Exception e3(e);
      REQUIRE(e3.alreadyPrinted());
    }
  }
  SECTION("message banner") {
    cms::Exception e1("ABC");
    const std::string expected("An exception of category 'ABC' occurred.\n");
    REQUIRE(e1.explainSelf() == expected);
  }
  SECTION("consistent message") {
    cms::Exception e1("ABC");
    cms::Exception e1s("ABC");
    REQUIRE(e1.explainSelf() == e1s.explainSelf());
  }

  SECTION("extend message") {
    cms::Exception e2("ABC", "foo");
    cms::Exception e2cs("ABC", std::string("foo"));
    cms::Exception e2sc(std::string("ABC"), "foo");
    cms::Exception e2ss(std::string("ABC"), std::string("foo"));
    e2 << "bar";
    e2cs << "bar";
    e2sc << "bar";
    e2ss << "bar";
    {
      const std::string expected(
          "An exception of category 'ABC' occurred.\n"
          "Exception Message:\n"
          "foo bar\n");
      REQUIRE(e2.explainSelf() == expected);
    }
    REQUIRE(e2.explainSelf() == e2cs.explainSelf());
    REQUIRE(e2.explainSelf() == e2sc.explainSelf());
    REQUIRE(e2.explainSelf() == e2ss.explainSelf());

    SECTION("partial message ends with space") {
      cms::Exception e3("ABC", "foo ");
      e3 << "bar\n";
      const std::string expected(
          "An exception of category 'ABC' occurred.\n"
          "Exception Message:\n"
          "foo bar\n");
      REQUIRE(e3.explainSelf() == expected);
    }
    SECTION("partial message ends with new line") {
      cms::Exception e4("ABC", "foo\n");
      e4 << "bar";
      const std::string expected(
          "An exception of category 'ABC' occurred.\n"
          "Exception Message:\n"
          "foo\nbar\n");
      REQUIRE(e4.explainSelf() == expected);
    }
  }
  SECTION("addContext") {
    cms::Exception e2("ABC", "foo bar");

    e2.addContext("context1");
    e2.addContext(std::string("context2"));
    e2.addAdditionalInfo("info1");
    e2.addAdditionalInfo(std::string("info2"));

    const std::string expected(
        "An exception of category 'ABC' occurred while\n"
        "   [0] context2\n"
        "   [1] context1\n"
        "Exception Message:\n"
        "foo bar \n"
        "   Additional Info:\n"
        "      [a] info2\n"
        "      [b] info1\n");
    REQUIRE(e2.explainSelf() == expected);
    SECTION("constructor message from other exception") {
      cms::Exception e6("DEF", "start", e2);
      e6 << "finish";
      std::string expected5(
          "An exception of category 'DEF' occurred while\n"
          "   [0] context2\n"
          "   [1] context1\n"
          "Exception Message:\n"
          "start\n"
          "foo bar "
          "finish\n"
          "   Additional Info:\n"
          "      [a] info2\n"
          "      [b] info1\n");
      REQUIRE(e6.explainSelf() == expected5);
      SECTION("copy constructor") {
        cms::Exception e7(e6);
        REQUIRE(e7.explainSelf() == expected5);
        REQUIRE(e7.category() == std::string("DEF"));
        REQUIRE(e7.message() == std::string("start\n"
                                            "foo bar "
                                            "finish"));
      }
      SECTION("clearContext") {
        e6.clearContext();
        std::string expected7_1(
            "An exception of category 'DEF' occurred.\n"
            "Exception Message:\n"
            "start\n"
            "foo bar "
            "finish\n"
            "   Additional Info:\n"
            "      [a] info2\n"
            "      [b] info1\n");
        REQUIRE(e6.explainSelf() == expected7_1);
      }
      SECTION("setContext") {
        std::list<std::string> newContext;
        newContext.push_back("new1");
        newContext.push_back("new2");
        newContext.push_back("new3");
        e6.setContext(newContext);
        REQUIRE(e6.context() == newContext);
      }
      SECTION("clearAdditionalInfo") {
        e6.clearAdditionalInfo();
        std::string expected7_2(
            "An exception of category 'DEF' occurred while\n"
            "   [0] context2\n"
            "   [1] context1\n"
            "Exception Message:\n"
            "start\n"
            "foo bar "
            "finish\n");
        REQUIRE(e6.explainSelf() == expected7_2);
      }
      SECTION("setAdditionalInfo") {
        std::list<std::string> newAdditionalInfo;
        newAdditionalInfo.push_back("newInfo1");
        newAdditionalInfo.push_back("newInfo2");
        newAdditionalInfo.push_back("newInfo3");
        e6.setAdditionalInfo(newAdditionalInfo);
        REQUIRE(e6.additionalInfo() == newAdditionalInfo);
        std::string expected7_3(
            "An exception of category 'DEF' occurred while\n"
            "   [0] context2\n"
            "   [1] context1\n"
            "Exception Message:\n"
            "start\n"
            "foo bar "
            "finish\n"
            "   Additional Info:\n"
            "      [a] newInfo3\n"
            "      [b] newInfo2\n"
            "      [c] newInfo1\n");
        REQUIRE(e6.explainSelf() == expected7_3);
      }
    }
  }

  cms::Exception e6("DEF", "start\nfoo barfinish");
  e6.setContext({{"new1", "new2", "new3"}});
  e6.setAdditionalInfo({"newInfo1", "newInfo2", "newInfo3"});
  SECTION("append") {
    e6.append(std::string(" X"));
    e6.append("Y");
    cms::Exception e8("ZZZ", "Z");
    e6.append(e8);
    std::string expected7_4(
        "An exception of category 'DEF' occurred while\n"
        "   [0] new3\n"
        "   [1] new2\n"
        "   [2] new1\n"
        "Exception Message:\n"
        "start\n"
        "foo bar"
        "finish  XYZ \n"
        "   Additional Info:\n"
        "      [a] newInfo3\n"
        "      [b] newInfo2\n"
        "      [c] newInfo1\n");
    REQUIRE(e6.explainSelf() == expected7_4);
  }
  SECTION("clearMessage") {
    e6.clearMessage();
    std::string expected7_5(
        "An exception of category 'DEF' occurred while\n"
        "   [0] new3\n"
        "   [1] new2\n"
        "   [2] new1\n"
        "   Additional Info:\n"
        "      [a] newInfo3\n"
        "      [b] newInfo2\n"
        "      [c] newInfo1\n");
    REQUIRE(e6.explainSelf() == expected7_5);
  }
  SECTION("raise") {
    std::unique_ptr<cms::Exception> ptr(e6.clone());
    std::string expected7_6(
        "An exception of category 'DEF' occurred while\n"
        "   [0] new3\n"
        "   [1] new2\n"
        "   [2] new1\n"
        "Exception Message:\n"
        "start\n"
        "foo bar"
        "finish \n"
        "   Additional Info:\n"
        "      [a] newInfo3\n"
        "      [b] newInfo2\n"
        "      [c] newInfo1\n");
    REQUIRE_THROWS_WITH(ptr->raise(), expected7_6);
  }
}
