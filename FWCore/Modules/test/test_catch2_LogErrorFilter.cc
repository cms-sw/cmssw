#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <vector>
#include "catch.hpp"

static constexpr auto s_tag = "[LogErrorFilter]";
TEST_CASE("Tests of LogErrorFilter", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.errorFilter = cms.EDFilter('LogErrorFilter',
                      harvesterTag = cms.InputTag('in'),
                      atLeastOneError = cms.bool(True),
                      atLeastOneWarning = cms.bool(True),
                      useThresholdsPerKind = cms.bool(False),
                      maxErrorKindsPerLumi = cms.uint32(1),
                      maxWarningKindsPerLumi = cms.uint32(1),
                      avoidCategories = cms.vstring()
)

process.moduleToTest(process.errorFilter)
)_"};

  SECTION("No threshhold, both warnings and errors") {
    edm::test::TestProcessor::Config config{baseConfig};

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  }

  SECTION("No threshhold, just errors") {
    edm::test::TestProcessor::Config config{baseConfig +
                                            R"_(
process.errorFilter.atLeastOneWarning = False
)_"};

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  }

  SECTION("No threshhold, just warnings") {
    edm::test::TestProcessor::Config config{baseConfig +
                                            R"_(
process.errorFilter.atLeastOneError = False
)_"};

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  }

  SECTION("No threshhold, errors, warnings, and categories") {
    edm::test::TestProcessor::Config config{baseConfig +
                                            R"_(
process.errorFilter.avoidCategories = ["IgnoreCat"]
)_"};

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_error},
                      {"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  };
  SECTION("threshhold of 1, both warnings and errors") {
    edm::test::TestProcessor::Config config{baseConfig +
                                            R"_(
process.errorFilter.useThresholdsPerKind = True
)_"

    };

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat2", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    tester.setLuminosityBlockNumber(2);
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat2", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  }

  SECTION("threshhold of 1, both warnings and errors, and categories") {
    edm::test::TestProcessor::Config config{baseConfig +
                                            R"_(
process.errorFilter.useThresholdsPerKind = True
process.errorFilter.avoidCategories = ["IgnoreCat"]
)_"

    };

    using ESVec = std::vector<edm::ErrorSummaryEntry>;
    auto const putToken = config.produces<ESVec>("in");

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>())).modulePassed());

    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_warning}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat2", "mod", edm::ELseverityLevel::ELsev_error}};
      REQUIRE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_fwkInfo}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"IgnoreCat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
    {
      ESVec errors = {{"Cat", "mod", edm::ELseverityLevel::ELsev_info}};
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
      REQUIRE_FALSE(tester.test(std::make_pair(putToken, std::make_unique<ESVec>(errors))).modulePassed());
    }
  }

  SECTION("Missing data") {
    edm::test::TestProcessor::Config config{baseConfig};
    edm::test::TestProcessor tester(config);

    REQUIRE_FALSE(tester.test().modulePassed());
  }
}
