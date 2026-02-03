#include <catch2/catch_all.hpp>

#include "FWCore/MessageLogger/interface/xmlUtils.h"

#include <sstream>

TEST_CASE("addElement", "[xmlUtils]") {
  SECTION("string") {
    std::ostringstream oss;
    edm::xml::addElement("PFN", "foo/bar", "", oss);
    REQUIRE(oss.str() == "<PFN>foo/bar</PFN>");
  }

  SECTION("integer") {
    std::ostringstream oss;
    edm::xml::addElement("PFN", 42, "\n", oss);
    REQUIRE(oss.str() == "<PFN>42</PFN>\n");
  }

  SECTION("floating point") {
    std::ostringstream oss;
    edm::xml::addElement("PFN", 3.14159, "\n", oss);
    REQUIRE(oss.str() == "<PFN>3.14159</PFN>\n");
  }

  SECTION("escapes special characters") {
    SECTION("ampersand") {
      std::ostringstream oss;
      edm::xml::addElement("PFN", "file?with&cgi=params", "\n", oss);
      REQUIRE(oss.str() == "<PFN>file?with&amp;cgi=params</PFN>\n");
    }
    SECTION("less and greater than") {
      std::ostringstream oss;
      edm::xml::addElement("Message", "file<other>file", "\n", oss);
      REQUIRE(oss.str() == "<Message>file&lt;other&gt;file</Message>\n");
    }
  }
}
