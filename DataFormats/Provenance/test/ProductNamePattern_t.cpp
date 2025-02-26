#include <cassert>
#include <iostream>

#include <catch.hpp>

#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {

  template <typename T>
  edm::ProductDescription make_ProductDescription(std::string const& module,
                                                  std::string const& instance,
                                                  std::string const& process) {
    edm::TypeWithDict type(typeid(T));
    edm::ProductDescription prod(edm::InEvent, module, process, type.name(), type.friendlyClassName(), instance, type);
    //prod.write(std::cerr);
    return prod;
  }

}  // namespace

TEST_CASE("test ProductNamePattern", "[ProductNamePattern]") {
  // int_module__TEST
  edm::ProductDescription prod1 = make_ProductDescription<int>("module", "", "TEST");
  // int_other__TEST
  edm::ProductDescription prod2 = make_ProductDescription<int>("other", "", "TEST");
  // int_module_label_TEST
  edm::ProductDescription prod3 = make_ProductDescription<int>("module", "label", "TEST");
  // int_module__PROC
  edm::ProductDescription prod4 = make_ProductDescription<int>("module", "", "PROC");
  // double_module__TEST
  edm::ProductDescription prod5 = make_ProductDescription<double>("module", "", "TEST");
  // Only transient products may have additional underscores in the module label.
  CHECK_THROWS(make_ProductDescription<bool>("transient_module", "", "TEST"));
  // edmtestTransientIntProduct_transient_module__TEST
  edm::ProductDescription prod6 = make_ProductDescription<edmtest::TransientIntProduct>("transient_module", "", "TEST");

  SECTION("Invalid pattern \"\"") {
    // An empty pattern is invalid.
    CHECK_THROWS_AS(edm::ProductNamePattern(""), edm::Exception);
  }

  SECTION("Simple pattern \"module\"") {
    // A simple pattern consisting only of a single field, without any underscores, is interpreted as a module label.
    edm::ProductNamePattern pattern("module");
    CHECK(pattern.match(prod1));
    CHECK(not pattern.match(prod2));
    CHECK(pattern.match(prod3));
    CHECK(pattern.match(prod4));
    CHECK(pattern.match(prod5));
    CHECK(not pattern.match(prod6));
  }

  SECTION("Invalid pattern \"transient_module\"") {
    // A pattern with a single underscore is invalid.
    CHECK_THROWS_AS(edm::ProductNamePattern("transient_module"), edm::Exception);
  }

  SECTION("Invalid pattern \"transient_module_label\"") {
    // A pattern with two underscores is invalid.
    CHECK_THROWS_AS(edm::ProductNamePattern("transient_module_label"), edm::Exception);
  }

  SECTION("Full branch pattern \"int_module__TEST\"") {
    // A full branch pattern must contain four fields, separated by an underscore.
    edm::ProductNamePattern pattern("int_module__TEST");
    CHECK(pattern.match(prod1));
    CHECK(not pattern.match(prod2));
    CHECK(not pattern.match(prod3));
    CHECK(not pattern.match(prod4));
    CHECK(not pattern.match(prod5));
    CHECK(not pattern.match(prod6));
  }

  SECTION("Simple wildcard \"*\"") {
    // A single * is interpreted as a wildcard that matches all branches.
    edm::ProductNamePattern pattern("*");
    CHECK(pattern.match(prod1));
    CHECK(pattern.match(prod2));
    CHECK(pattern.match(prod3));
    CHECK(pattern.match(prod4));
    CHECK(pattern.match(prod5));
    CHECK(pattern.match(prod6));
  }

  SECTION("Full branch wildcard \"*_*_*_*\"") {
    // Individual fields can be represented by a wildcard.
    edm::ProductNamePattern pattern("*_*_*_*");
    CHECK(pattern.match(prod1));
    CHECK(pattern.match(prod2));
    CHECK(pattern.match(prod3));
    CHECK(pattern.match(prod4));
    CHECK(pattern.match(prod5));
    CHECK(pattern.match(prod6));
  }

  SECTION("Branch wildcard with module label \"*_module_*_*\"") {
    // Individual fields can be represented by a wildcard.
    edm::ProductNamePattern pattern("*_module_*_*");
    CHECK(pattern.match(prod1));
    CHECK(not pattern.match(prod2));
    CHECK(pattern.match(prod3));
    CHECK(pattern.match(prod4));
    CHECK(pattern.match(prod5));
    CHECK(not pattern.match(prod6));
  }

  SECTION("Branch wildcard with module label \"*_transient_module_*_*\"") {
    // The module label may contain additional underscores.
    edm::ProductNamePattern pattern("*_transient_module_*_*");
    CHECK(not pattern.match(prod1));
    CHECK(not pattern.match(prod2));
    CHECK(not pattern.match(prod3));
    CHECK(not pattern.match(prod4));
    CHECK(not pattern.match(prod5));
    CHECK(pattern.match(prod6));
  }

  SECTION("Branch wildcard with process name \"*_*_*_TEST\"") {
    // Individual fields can be represented by a wildcard.
    edm::ProductNamePattern pattern("*_*_*_TEST");
    CHECK(pattern.match(prod1));
    CHECK(pattern.match(prod2));
    CHECK(pattern.match(prod3));
    CHECK(not pattern.match(prod4));
    CHECK(pattern.match(prod5));
    CHECK(pattern.match(prod6));
  }
}
