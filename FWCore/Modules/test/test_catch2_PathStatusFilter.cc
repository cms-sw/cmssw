#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/PathStatus.h"

#include <memory>
#include <utility>

static constexpr auto s_tag = "[PathStatusFilter]";

TEST_CASE("Standard checks of PathStatusFilter", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDFilter("PathStatusFilter")
process.moduleToTest(process.toTest)
)_"
  };

  SECTION("module constructor") {
    edm::test::TestProcessor::Config config{ baseConfig };
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }

  SECTION("No event data, empty expression") {
    edm::test::TestProcessor::Config config{ baseConfig };
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.test());

    auto event = tester.test();
    REQUIRE(event.modulePassed());
  }

  SECTION("Only a single path in expression") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("pathname")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token = testConfig.produces<edm::PathStatus>("pathname");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(event.modulePassed());

    event = tester.test(std::make_pair(token,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(!event.modulePassed());

    REQUIRE_THROWS(tester.test());
  }

  SECTION("test the operator 'and'") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("path1 and path2")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
  }

  SECTION("test the operator 'or'") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("path1 or path2")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
  }

  SECTION("test operator 'not'") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("not pathname")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token = testConfig.produces<edm::PathStatus>("pathname");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(!event.modulePassed());

    event = tester.test(std::make_pair(token,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(event.modulePassed());
  }

  SECTION("test precedence") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("path1 or path2 and path3")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    auto token3 = testConfig.produces<edm::PathStatus>("path3");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
  }
  SECTION("test parenthesis") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("(path1 or path2) and path3")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    auto token3 = testConfig.produces<edm::PathStatus>("path3");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
  }

  SECTION("test extra space between pathnames and operators") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("\t    path1   \t and \t  path2 \t and \t not(path3)and not not not path4 or(path5)and  not (  path6 or(path7)) \t   ")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    auto token3 = testConfig.produces<edm::PathStatus>("path3");
    auto token4 = testConfig.produces<edm::PathStatus>("path4");
    auto token5 = testConfig.produces<edm::PathStatus>("path5");
    auto token6 = testConfig.produces<edm::PathStatus>("path6");
    auto token7 = testConfig.produces<edm::PathStatus>("path7");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail))
    );
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token4,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token5,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token6,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                             std::make_pair(token7,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass))
    );
    REQUIRE(!event.modulePassed());
  }

  SECTION("test single character pathname") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("a")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token = testConfig.produces<edm::PathStatus>("a");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(event.modulePassed());

    event = tester.test(std::make_pair(token,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(!event.modulePassed());
  }

  SECTION("test pathnames containing an operator name and duplicate pathnames") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("nota and nota and anot")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("nota");
    auto token2 = testConfig.produces<edm::PathStatus>("anot");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(!event.modulePassed());
  }

  SECTION("test multiple parentheses and 'not's") {
    std::string fullConfig = baseConfig +
                             R"_(process.toTest.logicalExpression = cms.string("not not not (((not(not(((((not not path1))) or path2))) and path3)))")
                             )_";
    edm::test::TestProcessor::Config testConfig{ fullConfig };
    auto token1 = testConfig.produces<edm::PathStatus>("path1");
    auto token2 = testConfig.produces<edm::PathStatus>("path2");
    auto token3 = testConfig.produces<edm::PathStatus>("path3");
    edm::test::TestProcessor tester(testConfig);
    auto event = tester.test(std::make_pair(token1,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token2,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                             std::make_pair(token3,
                                            std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(!event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Pass)));
    REQUIRE(event.modulePassed());
    event = tester.test(std::make_pair(token1,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token2,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)),
                        std::make_pair(token3,
                                       std::make_unique<edm::PathStatus>(edm::hlt::Fail)));
    REQUIRE(event.modulePassed());
  }
}
