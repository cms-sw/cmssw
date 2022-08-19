#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

namespace {

  bool testExpression(std::string const& expression) {
    auto const* expr = triggerExpression::parse(expression);

    if (not expr) {
      edm::LogWarning("InvalidInput") << "Couldn't parse trigger results expression \"" << expression << "\"";
      return false;
    }

    edm::LogPrint("testExpression") << "Parsed expression: \"" << expression << "\"";
    return true;
  }

}  // namespace

TEST_CASE("Test TriggerExpressionParser", "[TriggerExpressionParser]") {
  // examples of expressions supported by the triggerExpression parser
  SECTION("CorrectExpressions") {
    REQUIRE(testExpression("TRUE"));
    REQUIRE(testExpression("FALSE"));
    REQUIRE(testExpression("NOT (FALSE)"));
    REQUIRE(testExpression("(NOT FALSE) OR TRUE"));
    REQUIRE(testExpression("NOTThisHLTPath AND TRUE AND NOT L1_A?_*"));
    REQUIRE(testExpression("NOT NOTThisHLTPath"));
    REQUIRE(testExpression("ThisHLTANDNOTThatORTheOther"));
    REQUIRE(testExpression("NOT L1_SEED1 AND L1_SEED2*"));
    REQUIRE(testExpression("NOT L1_SEED2 AND (HLT_PATH_? AND NOT HLT_PATH2_??_*)"));
    REQUIRE(testExpression("NOT (HLT_Path1 AND HLT_Path2)"));
    REQUIRE(testExpression("NOT (NOTHLT_Path OR HLT_Path2)"));
    REQUIRE(testExpression("((L1_A AND HLT_B) OR Dataset_C) AND NOT (Status_D OR Name_E OR HLT_F*) AND L1_??_?_?"));
    REQUIRE(testExpression("NOT (NOT (HLT_Path1 AND HLT_Path_*))"));
  }

  // examples of expressions not supported by the triggerExpression parser
  SECTION("IncorrectExpressions") {
    REQUIRE(not testExpression("A | B"));
    REQUIRE(not testExpression("A && B"));
    REQUIRE(not testExpression("NOT L1_SEED1 ANDD L1_SEED2*"));
    REQUIRE(not testExpression("NOT (NOTHLT_Path OR HLT_Path2))"));
    REQUIRE(not testExpression("NOT NOT (HLT_Path1 AND L1_Seed_?? OR HLT_Path_*)"));
    REQUIRE(not testExpression("HLT_Path* NOT TRUE"));
  }
}
