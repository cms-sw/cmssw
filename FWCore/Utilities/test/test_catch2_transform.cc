/*----------------------------------------------------------------------

Test program for edm::vector_transform class.

 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/transform.h"

namespace {
  std::string byvalue_toupper(std::string const& value) { return boost::to_upper_copy(value); }

  const std::string byconstvalue_toupper(std::string const& value) { return boost::to_upper_copy(value); }

  std::string& byref_toupper(std::string const& value) { return *new std::string(boost::to_upper_copy(value)); }

  std::string& byconstref_toupper(std::string const& value) { return *new std::string(boost::to_upper_copy(value)); }
}  // namespace

TEST_CASE("edm::vector_transform", "[transform]") {
  const std::vector<std::string> input{"Hello", "World"};
  const std::vector<std::string> upper{"HELLO", "WORLD"};
  const std::vector<std::string::size_type> size{5, 5};

  auto test_lambda = edm::vector_transform(input, [](std::string const& value) { return value.size(); });
  REQUIRE(size == test_lambda);

  REQUIRE(upper == edm::vector_transform(input, byvalue_toupper));
  REQUIRE(upper == edm::vector_transform(input, byconstvalue_toupper));
  REQUIRE(upper == edm::vector_transform(input, byref_toupper));
  REQUIRE(upper == edm::vector_transform(input, byconstref_toupper));
}
