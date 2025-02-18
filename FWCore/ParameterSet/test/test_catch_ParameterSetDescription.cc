
// Test code for the ParameterSetDescription and ParameterDescription
// classes.

#include "catch.hpp"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"
#include "FWCore/ParameterSet/interface/IfExistsDescription.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterWildcardWithSpecifics.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace testParameterSetDescription {

  void testDesc(edm::ParameterDescriptionNode const& node,
                edm::ParameterSetDescription const& psetDesc,
                edm::ParameterSet& pset,
                bool exists,
                bool validates) {
    CHECK(node.exists(pset) == exists);
    CHECK(node.partiallyExists(pset) == exists);
    CHECK(node.howManyXORSubNodesExist(pset) == (exists ? 1 : 0));
    if (validates) {
      psetDesc.validate(pset);
    } else {
      REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
    }
  }

  struct TestPluginBase {
    virtual ~TestPluginBase() = default;
  };

  struct ATestPlugin : public TestPluginBase {
    static void fillPSetDescription(edm::ParameterSetDescription& iPS) { iPS.add<int>("anInt", 5); }
  };

  struct BTestPlugin : public TestPluginBase {
    static void fillPSetDescription(edm::ParameterSetDescription& iPS) { iPS.add<double>("aDouble", 0.5); }
  };

  using TestPluginFactory = edmplugin::PluginFactory<testParameterSetDescription::TestPluginBase*()>;

}  // namespace testParameterSetDescription

using TestPluginFactory = testParameterSetDescription::TestPluginFactory;

using testParameterSetDescription::testDesc;

TEST_CASE("test ParameterSetDescription", "[ParameterSetDescription]") {
  SECTION("testWildcards") {
    using Catch::Matchers::Equals;
    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<int> w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<int>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<int>("y", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<unsigned>("z", 1);
      testDesc(w, set, pset, true, false);

      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};
        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(), Equals("\nallowAnyLabel_ = cms.required.int32"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireExactlyOne, false);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, false, false);

      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};

        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(), Equals("\nallowAnyLabel_ = cms.required.untracked.uint32"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireExactlyOne, false);
      set.addOptionalNode(w, false);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, true);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<unsigned> w("*", edm::RequireAtLeastOne, false);
      set.addOptionalNode(w, false);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, true);
      pset.addUntrackedParameter<unsigned>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addUntrackedParameter<unsigned>("y", 1);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<double>("x", 1);
      testDesc(w, set, pset, true, true);
      pset.addParameter<double>("y", 1);
      testDesc(w, set, pset, true, true);

      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};

        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(), Equals("\nallowAnyLabel_ = cms.required.double"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
      set.addNode(w);
      set.add<int>("testTypeChecking1", 11);
      REQUIRE_THROWS_AS(set.add<double>("testTypeChecking2", 11.0), edm::Exception);
    }

    REQUIRE_THROWS_AS(edm::ParameterWildcard<int>("a*", edm::RequireZeroOrMore, true), edm::Exception);

    edm::ParameterSet nestedPset;
    nestedPset.addUntrackedParameter<unsigned>("n1", 1);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);

      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};
        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(), Equals("\nallowAnyLabel_ = cms.required.PSetTemplate()"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w(std::string("*"), edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w(
          "*", edm::RequireZeroOrMore, true, edm::ParameterSetDescription());
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, false);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, false);
    }

    edm::ParameterSetDescription nestedDesc;
    nestedDesc.addUntracked<unsigned>("n1", 1);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireZeroOrMore, true, nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);

      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};
        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(),
                     Equals("\nallowAnyLabel_ = cms.required.PSetTemplate(\n  n1 = cms.untracked.uint32(1)\n)"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w(
          std::string("*"), edm::RequireExactlyOne, true, nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireAtLeastOne, true, nestedDesc);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<edm::ParameterSet>("nested1", nestedPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<edm::ParameterSet>("nested2", nestedPset);
      testDesc(w, set, pset, true, true);
    }

    std::vector<edm::ParameterSet> nestedVPset;
    edm::ParameterSet vectorElement;
    vectorElement.addUntrackedParameter<unsigned>("n11", 1);
    nestedVPset.push_back(vectorElement);
    nestedVPset.push_back(vectorElement);

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w("*", edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
      SECTION("cfi generation") {
        std::ostringstream os;
        bool startWithComma = false;
        bool wroteSomething = false;
        edm::CfiOptions ops = edm::cfi::Typed{};
        w.writeCfi(os, false, startWithComma, 0, ops, wroteSomething);

        REQUIRE_THAT(os.str(), Equals("\nallowAnyLabel_ = cms.required.VPSet"));
      }
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w(std::string("*"), edm::RequireZeroOrMore, true);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w(
          "*", edm::RequireZeroOrMore, true, edm::ParameterSetDescription());
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, false);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, true, false);
    }

    edm::ParameterSetDescription descElement;
    descElement.addUntracked<unsigned>("n11", 1);
    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w("*", edm::RequireZeroOrMore, true, descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w(
          std::string("*"), edm::RequireExactlyOne, true, descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, false, false);
    }

    {
      edm::ParameterSetDescription set;
      edm::ParameterWildcard<std::vector<edm::ParameterSet>> w("*", edm::RequireAtLeastOne, true, descElement);
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, false, false);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested1", nestedVPset);
      testDesc(w, set, pset, true, true);
      pset.addParameter<std::vector<edm::ParameterSet>>("nested2", nestedVPset);
      testDesc(w, set, pset, true, true);
    }
  }

  SECTION("testWildcardWithExceptions") {
    {
      edm::ParameterSetDescription set;

      edm::ParameterSetDescription wild;
      wild.addUntracked<unsigned>("n11", 1);

      edm::ParameterSetDescription except_;
      except_.addUntracked<double>("f", 3.14);
      std::map<std::string, edm::ParameterSetDescription> excptions = {{"special", except_}};
      edm::ParameterWildcardWithSpecifics w("*", edm::RequireZeroOrMore, true, wild, std::move(excptions));
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      edm::ParameterSet nested1;
      nested1.addUntrackedParameter<unsigned>("n11", 3);
      pset.addParameter<edm::ParameterSet>("nested1", nested1);
      testDesc(w, set, pset, true, true);
      edm::ParameterSet special;
      special.addUntrackedParameter<double>("f", 5);
      pset.addParameter<edm::ParameterSet>("special", special);
      testDesc(w, set, pset, true, true);
    }

    {
      edm::ParameterSetDescription set;

      edm::ParameterSetDescription wild;
      wild.add<unsigned>("n11", 1);

      edm::ParameterSetDescription except_;
      except_.add<double>("f", 3.14);
      std::map<std::string, edm::ParameterSetDescription> excptions = {{"special", except_}};
      edm::ParameterWildcardWithSpecifics w("*", edm::RequireZeroOrMore, true, wild, std::move(excptions));
      set.addNode(w);
      edm::ParameterSet pset;
      testDesc(w, set, pset, true, true);
      edm::ParameterSet nested1;
      nested1.addParameter<unsigned>("n11", 3);
      pset.addParameter<edm::ParameterSet>("nested1", nested1);
      testDesc(w, set, pset, true, true);
      edm::ParameterSet special;
      special.addParameter<double>("f", 5);
      pset.addParameter<edm::ParameterSet>("special", special);
      testDesc(w, set, pset, true, true);
    }
  }

  // ---------------------------------------------------------------------------------

  SECTION("testAllowedValues") {
    // Duplicate case values not allowed
    edm::ParameterSetDescription psetDesc;
    psetDesc.ifValue(edm::ParameterDescription<std::string>("sswitch", "a", true),
                     edm::allowedValues<std::string>("a", "h", "z"));
  }

  SECTION("testSwitch") {
    // Duplicate case values not allowed
    edm::ParameterSetDescription psetDesc;
    REQUIRE_THROWS_AS(psetDesc.ifValue(edm::ParameterDescription<int>("oiswitch", 1, true),
                                       0 >> edm::ParameterDescription<int>("oivalue", 100, true) or
                                           1 >> (edm::ParameterDescription<double>("oivalue1", 101.0, true) and
                                                 edm::ParameterDescription<double>("oivalue2", 101.0, true)) or
                                           1 >> edm::ParameterDescription<std::string>("oivalue", "102", true)),
                      edm::Exception);

    // Types used in case parameters cannot duplicate type already used in a wildcard
    edm::ParameterSetDescription psetDesc1;
    edm::ParameterWildcard<double> w("*", edm::RequireAtLeastOne, true);
    psetDesc1.addNode(w);

    REQUIRE_THROWS_AS(psetDesc1.ifValue(edm::ParameterDescription<int>("oiswitch", 1, true),
                                        0 >> edm::ParameterDescription<int>("oivalue", 100, true) or
                                            1 >> (edm::ParameterDescription<double>("oivalue1", 101.0, true) and
                                                  edm::ParameterDescription<double>("oivalue2", 101.0, true)) or
                                            2 >> edm::ParameterDescription<std::string>("oivalue", "102", true)),
                      edm::Exception);

    // Types used in the switch parameter cannot duplicate type already used in a wildcard
    edm::ParameterSetDescription psetDesc2;
    edm::ParameterWildcard<int> w1("*", edm::RequireAtLeastOne, true);
    psetDesc2.addNode(w1);

    REQUIRE_THROWS_AS(psetDesc2.ifValue(edm::ParameterDescription<int>("aswitch", 1, true),
                                        1 >> (edm::ParameterDescription<unsigned>("avalue1", 101, true) and
                                              edm::ParameterDescription<unsigned>("avalue2", 101, true)) or
                                            2 >> edm::ParameterDescription<std::string>("avalue", "102", true)),
                      edm::Exception);

    // Type used in the switch parameter cannot duplicate type in a case wildcard
    edm::ParameterSetDescription psetDesc3;
    REQUIRE_THROWS_AS(psetDesc3.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                                        0 >> edm::ParameterWildcard<int>("*", edm::RequireAtLeastOne, true) or
                                            1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                                  edm::ParameterDescription<double>("xvalue2", 101.0, true))),
                      edm::Exception);

    // Type used in a parameter cannot duplicate type in a case wildcard
    edm::ParameterSetDescription psetDesc4;
    psetDesc4.add<unsigned>("testunsigned", 1U);
    REQUIRE_THROWS_AS(psetDesc4.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                                        0 >> edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true) or
                                            1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                                  edm::ParameterDescription<double>("xvalue2", 101.0, true))),
                      edm::Exception);

    // No problem is wildcard type and parameter type are the same for different cases.
    edm::ParameterSetDescription psetDesc5;
    psetDesc5.ifValue(edm::ParameterDescription<int>("uswitch", 1, true),
                      0 >> edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true) or
                          1 >> (edm::ParameterDescription<unsigned>("uvalue1", 101, true) and
                                edm::ParameterDescription<unsigned>("uvalue2", 101, true)));

    // The switch parameter label cannot be the same as a label that already exists
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<unsigned>("xswitch", 1U);
    REQUIRE_THROWS_AS(psetDesc6.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                                        0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                                            1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                                  edm::ParameterDescription<double>("xvalue2", 101.0, true))),
                      edm::Exception);

    // Case labels cannot be the same as a label that already exists
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.add<unsigned>("xvalue1", 1U);
    REQUIRE_THROWS_AS(psetDesc7.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                                        0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                                            1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                                  edm::ParameterDescription<double>("xvalue2", 101.0, true))),
                      edm::Exception);

    // Case labels cannot be the same as a switch label
    edm::ParameterSetDescription psetDesc8;
    REQUIRE_THROWS_AS(psetDesc8.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                                        0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                                            1 >> (edm::ParameterDescription<double>("xswitch", 101.0, true) and
                                                  edm::ParameterDescription<double>("xvalue2", 101.0, true))),
                      edm::Exception);

    // Parameter set switch value must be one of the defined cases
    edm::ParameterSetDescription psetDesc9;

    psetDesc9.ifValue(edm::ParameterDescription<int>("xswitch", 1, true),
                      0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                          1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                edm::ParameterDescription<double>("xvalue2", 101.0, true)));
    edm::ParameterSet pset;
    pset.addParameter<int>("xswitch", 5);
    REQUIRE_THROWS_AS(psetDesc9.validate(pset), edm::Exception);

    edm::ParameterSwitch<int> pswitch(edm::ParameterDescription<int>("xswitch", 1, true),
                                      0 >> edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
                                          1 >> (edm::ParameterDescription<double>("xvalue1", 101.0, true) and
                                                edm::ParameterDescription<double>("xvalue2", 101.0, true)));
    edm::ParameterSetDescription psetDesc10;
    psetDesc10.addNode(pswitch);
    edm::ParameterSet pset10;
    testDesc(pswitch, psetDesc10, pset10, false, true);
    pset10.addParameter<int>("xswitch", 1);
    testDesc(pswitch, psetDesc10, pset10, true, true);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testXor") {
    edm::ParameterSetDescription psetDesc1;
    std::unique_ptr<edm::ParameterDescriptionNode> node1(
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) and
         edm::ParameterDescription<double>("x2", 101.0, true)) xor
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) or edm::ParameterDescription<double>("x2", 101.0, true)));

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);

    CHECK(node1->exists(pset1) == false);
    CHECK(node1->partiallyExists(pset1) == false);
    CHECK(node1->howManyXORSubNodesExist(pset1) == 0);

    CHECK(node1->exists(pset2) == false);
    CHECK(node1->partiallyExists(pset2) == false);
    CHECK(node1->howManyXORSubNodesExist(pset2) == 4);

    // 0 of the options existing should fail validation
    psetDesc1.addNode(std::move(node1));
    REQUIRE_THROWS_AS(psetDesc1.validate(pset1), edm::Exception);

    // More than one of the options existing should also fail
    REQUIRE_THROWS_AS(psetDesc1.validate(pset2), edm::Exception);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("xvalue1", 1U);
    std::unique_ptr<edm::ParameterDescriptionNode> node2(edm::ParameterDescription<double>("x1", 101.0, true) xor
                                                         (edm::ParameterDescription<double>("x1", 101.0, true) and
                                                          edm::ParameterDescription<double>("x2", 101.0, true)) xor
                                                         edm::ParameterDescription<double>("x1", 101.0, true) xor
                                                         (edm::ParameterDescription<double>("xvalue1", 101.0, true) or
                                                          edm::ParameterDescription<double>("x2", 101.0, true)));
    REQUIRE_THROWS_AS(psetDesc2.addNode(std::move(node2)), edm::Exception);

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::unique_ptr<edm::ParameterDescriptionNode> node3(edm::ParameterDescription<double>("x1", 101.0, true) xor
                                                         (edm::ParameterDescription<double>("x1", 101.0, true) and
                                                          edm::ParameterDescription<double>("x2", 101.0, true)) xor
                                                         edm::ParameterDescription<double>("x1", 101.0, true) xor
                                                         (edm::ParameterDescription<double>("xvalue1", 101.0, true) or
                                                          edm::ParameterDescription<double>("x2", 101.0, true)));
    psetDesc3.addNode(std::move(node3));
    REQUIRE_THROWS_AS(psetDesc3.add<unsigned>("xvalue1", 1U), edm::Exception);

    // A parameter cannot use the same type as a wildcard
    edm::ParameterSetDescription psetDesc4;
    std::unique_ptr<edm::ParameterDescriptionNode> node4(
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) and
         edm::ParameterDescription<unsigned>("x2", 101, true)) xor
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) or edm::ParameterDescription<double>("x2", 101.0, true)));
    psetDesc4.addNode(std::move(node4));

    edm::ParameterWildcard<unsigned> w4("*", edm::RequireAtLeastOne, true);
    REQUIRE_THROWS_AS(psetDesc4.addNode(w4), edm::Exception);

    // A parameter cannot use the same type as a wildcard
    edm::ParameterSetDescription psetDesc5;
    std::unique_ptr<edm::ParameterDescriptionNode> node5(
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) and
         edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true)) xor
        edm::ParameterDescription<double>("x1", 101.0, true) xor
        (edm::ParameterDescription<double>("x1", 101.0, true) or
         edm::ParameterWildcard<unsigned>("*", edm::RequireAtLeastOne, true)));
    psetDesc5.addNode(std::move(node5));

    edm::ParameterDescription<unsigned> n5("z5", edm::RequireAtLeastOne, true);
    REQUIRE_THROWS_AS(psetDesc5.addNode(n5), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testOr") {
    edm::ParameterSetDescription psetDesc1;
    std::unique_ptr<edm::ParameterDescriptionNode> node1(edm::ParameterDescription<double>("x1", 101.0, true) or
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) or
                                                         edm::ParameterDescription<double>("x4", 101.0, true) or
                                                         (edm::ParameterDescription<double>("x5", 101.0, true) xor
                                                          edm::ParameterDescription<double>("x6", 101.0, true)));

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);
    pset2.addParameter("x4", 14.0);
    pset2.addParameter("x5", 15.0);

    CHECK(node1->exists(pset1) == false);
    CHECK(node1->partiallyExists(pset1) == false);
    CHECK(node1->howManyXORSubNodesExist(pset1) == 0);

    CHECK(node1->exists(pset2) == true);
    CHECK(node1->partiallyExists(pset2) == true);
    CHECK(node1->howManyXORSubNodesExist(pset2) == 1);

    // 0 of the options existing should fail validation
    psetDesc1.addNode(std::move(node1));
    psetDesc1.validate(pset1);

    // More than one of the options existing should succeed
    psetDesc1.validate(pset2);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::unique_ptr<edm::ParameterDescriptionNode> node2(edm::ParameterDescription<double>("x1", 101.0, true) or
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) or
                                                         edm::ParameterDescription<double>("x4", 101.0, true));
    REQUIRE_THROWS_AS(psetDesc2.addNode(std::move(node2)), edm::Exception);

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::unique_ptr<edm::ParameterDescriptionNode> node3(edm::ParameterDescription<double>("x1", 101.0, true) or
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) or
                                                         edm::ParameterDescription<double>("x4", 101.0, true));
    psetDesc3.addNode(std::move(node3));

    REQUIRE_THROWS_AS(psetDesc3.add<unsigned>("x1", 1U), edm::Exception);

    // Put the duplicate labels in different nodes of the "or" expression
    edm::ParameterSetDescription psetDesc4;
    std::unique_ptr<edm::ParameterDescriptionNode> node4(edm::ParameterDescription<double>("x1", 101.0, true) or
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) or
                                                         edm::ParameterDescription<double>("x1", 101.0, true));
    REQUIRE_THROWS_AS(psetDesc4.addNode(std::move(node4)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::unique_ptr<edm::ParameterDescriptionNode> node5(
        edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<unsigned>("x3", 101U, true)) or
        edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc5.addNode(std::move(node5)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::unique_ptr<edm::ParameterDescriptionNode> node6(
        edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) or
        (edm::ParameterDescription<unsigned>("x2", 101U, true) and
         edm::ParameterDescription<unsigned>("x3", 101U, true)) or
        edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc6.addNode(std::move(node6)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::unique_ptr<edm::ParameterDescriptionNode> node7(edm::ParameterDescription<double>("x0", 1.0, true) or
                                                         (edm::ParameterDescription<unsigned>("x2", 101U, true) and
                                                          edm::ParameterDescription<unsigned>("x3", 101U, true)) or
                                                         edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc7.addNode(std::move(node7)), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testAnd") {
    edm::ParameterSetDescription psetDesc1;
    std::unique_ptr<edm::ParameterDescriptionNode> node1(edm::ParameterDescription<double>("x1", 101.0, true) and
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) or
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) and
                                                         edm::ParameterDescription<double>("x4", 101.0, true) and
                                                         (edm::ParameterDescription<double>("x5", 101.0, true) xor
                                                          edm::ParameterDescription<double>("x6", 101.0, true)));

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);
    pset2.addParameter("x4", 14.0);
    pset2.addParameter("x5", 15.0);

    edm::ParameterSet pset3;
    pset3.addParameter("x3", 13.0);

    CHECK(node1->exists(pset1) == false);
    CHECK(node1->partiallyExists(pset1) == false);
    CHECK(node1->howManyXORSubNodesExist(pset1) == 0);

    CHECK(node1->exists(pset2) == true);
    CHECK(node1->partiallyExists(pset2) == true);
    CHECK(node1->howManyXORSubNodesExist(pset2) == 1);

    CHECK(node1->exists(pset3) == false);
    CHECK(node1->partiallyExists(pset3) == true);
    CHECK(node1->howManyXORSubNodesExist(pset3) == 0);

    psetDesc1.addNode(std::move(node1));
    psetDesc1.validate(pset1);
    psetDesc1.validate(pset2);
    psetDesc1.validate(pset3);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::unique_ptr<edm::ParameterDescriptionNode> node2(edm::ParameterDescription<double>("x1", 101.0, true) and
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) or
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) and
                                                         edm::ParameterDescription<double>("x4", 101.0, true));
    REQUIRE_THROWS_AS(psetDesc2.addNode(std::move(node2)), edm::Exception);

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::unique_ptr<edm::ParameterDescriptionNode> node3(edm::ParameterDescription<double>("x1", 101.0, true) and
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) or
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) and
                                                         edm::ParameterDescription<double>("x4", 101.0, true));
    psetDesc3.addNode(std::move(node3));

    REQUIRE_THROWS_AS(psetDesc3.add<unsigned>("x1", 1U), edm::Exception);

    // Put the duplicate labels in different nodes of the "and" expression
    edm::ParameterSetDescription psetDesc4;
    std::unique_ptr<edm::ParameterDescriptionNode> node4(edm::ParameterDescription<double>("x1", 101.0, true) and
                                                         (edm::ParameterDescription<double>("x2", 101.0, true) or
                                                          edm::ParameterDescription<double>("x3", 101.0, true)) and
                                                         edm::ParameterDescription<double>("x1", 101.0, true));
    REQUIRE_THROWS_AS(psetDesc4.addNode(std::move(node4)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::unique_ptr<edm::ParameterDescriptionNode> node5(
        edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) and
        (edm::ParameterDescription<double>("x2", 101.0, true) or
         edm::ParameterDescription<unsigned>("x3", 101U, true)) and
        edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc5.addNode(std::move(node5)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::unique_ptr<edm::ParameterDescriptionNode> node6(
        edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true) and
        (edm::ParameterDescription<unsigned>("x2", 101U, true) or
         edm::ParameterDescription<unsigned>("x3", 101U, true)) and
        edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc6.addNode(std::move(node6)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::unique_ptr<edm::ParameterDescriptionNode> node7(edm::ParameterDescription<double>("x0", 1.0, true) and
                                                         (edm::ParameterDescription<unsigned>("x2", 101U, true) or
                                                          edm::ParameterDescription<unsigned>("x3", 101U, true)) and
                                                         edm::ParameterDescription<unsigned>("x1", 101U, true));
    REQUIRE_THROWS_AS(psetDesc7.addNode(std::move(node7)), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testIfExists") {
    edm::ParameterSetDescription psetDesc1;
    std::unique_ptr<edm::ParameterDescriptionNode> node1(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterDescription<double>("x1", 101.0, true),
                                                   (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                    edm::ParameterDescription<double>("x3", 101.0, true))));

    std::unique_ptr<edm::ParameterDescriptionNode> node1a(std::make_unique<edm::IfExistsDescription>(
        (edm::ParameterDescription<double>("x2", 101.0, true) and edm::ParameterDescription<double>("x3", 101.0, true)),
        edm::ParameterDescription<double>("x1", 101.0, true)));

    std::unique_ptr<edm::ParameterDescriptionNode> node1b(std::make_unique<edm::IfExistsDescription>(
        (edm::ParameterDescription<double>("x1", 101.0, true) xor edm::ParameterDescription<int>("x1", 101, true)),
        (edm::ParameterDescription<double>("x2", 101.0, true) and
         edm::ParameterDescription<double>("x3", 101.0, true))));

    edm::ParameterSet pset1;

    edm::ParameterSet pset2;
    pset2.addParameter("x1", 11.0);
    pset2.addParameter("x2", 12.0);
    pset2.addParameter("x3", 13.0);

    edm::ParameterSet pset3;
    pset3.addParameter("x1", 14.0);

    edm::ParameterSet pset4;
    pset4.addParameter("x2", 15.0);
    pset4.addParameter("x3", 16.0);

    CHECK(node1->exists(pset1) == true);
    CHECK(node1->partiallyExists(pset1) == true);
    CHECK(node1->howManyXORSubNodesExist(pset1) == 1);
    CHECK(node1a->exists(pset1) == true);
    CHECK(node1b->exists(pset1) == true);

    CHECK(node1->exists(pset2) == true);
    CHECK(node1->partiallyExists(pset2) == true);
    CHECK(node1->howManyXORSubNodesExist(pset2) == 1);
    CHECK(node1a->exists(pset2) == true);
    CHECK(node1b->exists(pset2) == true);

    CHECK(node1->exists(pset3) == false);
    CHECK(node1->partiallyExists(pset3) == false);
    CHECK(node1->howManyXORSubNodesExist(pset3) == 0);
    CHECK(node1a->exists(pset3) == false);
    CHECK(node1b->exists(pset3) == false);

    CHECK(node1->exists(pset4) == false);
    CHECK(node1->partiallyExists(pset4) == false);
    CHECK(node1->howManyXORSubNodesExist(pset4) == 0);
    CHECK(node1a->exists(pset4) == false);
    CHECK(node1b->exists(pset4) == false);

    psetDesc1.addNode(std::move(node1));
    psetDesc1.validate(pset1);
    psetDesc1.validate(pset2);
    psetDesc1.validate(pset3);
    psetDesc1.validate(pset3);

    // One of the labels cannot already exist in the description
    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<unsigned>("x1", 1U);
    std::unique_ptr<edm::ParameterDescriptionNode> node2(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterDescription<double>("x1", 101.0, true),
                                                   (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                    edm::ParameterDescription<double>("x3", 101.0, true))));

    REQUIRE_THROWS_AS(psetDesc2.addNode(std::move(node2)), edm::Exception);

    // One of the labels cannot already exist in the description, other order
    edm::ParameterSetDescription psetDesc3;
    std::unique_ptr<edm::ParameterDescriptionNode> node3(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterDescription<double>("x1", 101.0, true),
                                                   (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                    edm::ParameterDescription<double>("x3", 101.0, true))));
    psetDesc3.addNode(std::move(node3));

    REQUIRE_THROWS_AS(psetDesc3.add<unsigned>("x1", 1U), edm::Exception);

    // Put the duplicate labels in different nodes of the "and" expression
    edm::ParameterSetDescription psetDesc4;
    std::unique_ptr<edm::ParameterDescriptionNode> node4(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterDescription<double>("x1", 101.0, true),
                                                   (edm::ParameterDescription<double>("x2", 101.0, true) and
                                                    edm::ParameterDescription<double>("x1", 101.0, true))));
    REQUIRE_THROWS_AS(psetDesc4.addNode(std::move(node4)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter
    edm::ParameterSetDescription psetDesc5;
    std::unique_ptr<edm::ParameterDescriptionNode> node5(std::make_unique<edm::IfExistsDescription>(
        edm::ParameterDescription<double>("x1", 101.0, true),
        (edm::ParameterDescription<unsigned>("x2", 101U, true) and
         edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true))));
    REQUIRE_THROWS_AS(psetDesc5.addNode(std::move(node5)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc6;
    psetDesc6.add<double>("x0", 1.0);
    std::unique_ptr<edm::ParameterDescriptionNode> node6(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterWildcard<double>("*", edm::RequireAtLeastOne, true),
                                                   (edm::ParameterDescription<unsigned>("x2", 101U, true) and
                                                    edm::ParameterDescription<unsigned>("x3", 102U, true))));
    REQUIRE_THROWS_AS(psetDesc6.addNode(std::move(node6)), edm::Exception);

    // A type used in a wildcard should not be the same as a type
    // used for another parameter node
    edm::ParameterSetDescription psetDesc7;
    psetDesc7.addWildcard<double>("*");
    std::unique_ptr<edm::ParameterDescriptionNode> node7(
        std::make_unique<edm::IfExistsDescription>(edm::ParameterDescription<double>("x1", 11.0, true),
                                                   (edm::ParameterDescription<unsigned>("x2", 101U, true) and
                                                    edm::ParameterDescription<unsigned>("x3", 102U, true))));
    REQUIRE_THROWS_AS(psetDesc7.addNode(std::move(node7)), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testAllowedLabels") {
    {
      std::unique_ptr<edm::ParameterDescriptionNode> node(
          std::make_unique<edm::AllowedLabelsDescription<int>>("allowedLabels", true));

      const edm::ParameterSet emptyPset;

      edm::ParameterSet pset;
      std::vector<std::string> labels;
      pset.addParameter<std::vector<std::string>>("allowedLabels", labels);

      CHECK(node->exists(emptyPset) == false);
      CHECK(node->partiallyExists(emptyPset) == false);
      CHECK(node->howManyXORSubNodesExist(emptyPset) == 0);

      CHECK(node->exists(pset) == true);
      CHECK(node->partiallyExists(pset) == true);
      CHECK(node->howManyXORSubNodesExist(pset) == 1);
    }

    {
      // One of the labels cannot already exist in the description
      edm::ParameterSetDescription psetDesc;
      psetDesc.add<unsigned>("x1", 1U);
      std::unique_ptr<edm::ParameterDescriptionNode> node(
          std::make_unique<edm::AllowedLabelsDescription<int>>("x1", true));

      REQUIRE_THROWS_AS(psetDesc.addNode(std::move(node)), edm::Exception);
    }

    {
      // A type used in a wildcard should not be the same as a type
      // used for another parameter node
      edm::ParameterSetDescription psetDesc;
      psetDesc.addWildcard<std::vector<std::string>>("*");
      std::unique_ptr<edm::ParameterDescriptionNode> node(
          std::make_unique<edm::AllowedLabelsDescription<int>>("x1", true));
      REQUIRE_THROWS_AS(psetDesc.addNode(std::move(node)), edm::Exception);
    }
    {
      edm::ParameterSet pset;
      edm::ParameterSet nestedPset;
      nestedPset.addParameter<int>("x", 1);
      pset.addParameter<edm::ParameterSet>("nestedPset", nestedPset);

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA");

        // nestedPset is an illegal parameter
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);

        std::vector<std::string> labels;
        labels.push_back(std::string("nestedPset"));
        pset.addParameter<std::vector<std::string>>("allowedLabelsA", labels);

        // Now nestedPset should be an allowed parameter
        psetDesc.validate(pset);
      }

      // Above it did not validate the contents of the nested ParameterSet
      // because a description was not passed to the labelsFrom function.

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA", edm::ParameterSetDescription());
        // Now it should fail because the description says the nested ParameterSet
        // should be empty, but it has parameter "x"
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
      }

      // Now include "x" in the description and it should once again pass validation
      edm::ParameterSetDescription nestedPsetDesc;
      nestedPsetDesc.add<int>("x", 1);

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>("allowedLabelsA", nestedPsetDesc);
        psetDesc.validate(pset);
      }
      // Minor variations, repeat with a string argument
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"));
        psetDesc.validate(pset);
      }
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"),
                                                          edm::ParameterSetDescription());
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
      }
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<edm::ParameterSetDescription>(std::string("allowedLabelsA"), nestedPsetDesc);
        psetDesc.validate(pset);
      }
    }
    // Now repeat what was done above with the variations
    // necessary to test the vector<ParameterSet> case
    {
      edm::ParameterSet pset;

      edm::ParameterSet elementOfVPset;
      elementOfVPset.addParameter<int>("y", 1);
      std::vector<edm::ParameterSet> vpset;
      vpset.push_back(elementOfVPset);
      vpset.push_back(elementOfVPset);

      pset.addParameter<std::vector<edm::ParameterSet>>("nestedVPSet", vpset);

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>("allowedLabelsC");

        // nestedVPSet is an illegal parameter
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);

        std::vector<std::string> labels;
        labels.push_back(std::string("nestedVPSet"));
        pset.addParameter<std::vector<std::string>>("allowedLabelsC", labels);

        // Now nestedVPSet should be an allowed parameter
        psetDesc.validate(pset);
      }
      // Above it did not validate the contents of the nested vector<ParameterSet>
      // because a description was not passed to the labelsFrom function.

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>("allowedLabelsC", edm::ParameterSetDescription());
        // Now it should fail because the description says the contained vector<ParameterSet>
        // should have empty ParameterSets, but the ParameterSets have parameter "y"
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
      }

      // Now include "y" in the description and it should once again pass validation
      edm::ParameterSetDescription nestedPSetDesc;
      nestedPSetDesc.add<int>("y", 1);

      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>("allowedLabelsC", nestedPSetDesc);
        psetDesc.validate(pset);
      }

      // Minor variations, repeat with a string argument
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>(std::string("allowedLabelsC"));
        psetDesc.validate(pset);
      }
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>(std::string("allowedLabelsC"),
                                                            edm::ParameterSetDescription());
        REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
      }
      {
        edm::ParameterSetDescription psetDesc;
        psetDesc.labelsFrom<std::vector<edm::ParameterSet>>(std::string("allowedLabelsC"), nestedPSetDesc);
        psetDesc.validate(pset);
      }
    }
  }
  // ---------------------------------------------------------------------------------

  SECTION("testNoDefault") {
    edm::ParameterSetDescription psetDesc;
    psetDesc.add<int>("x");
    edm::ParameterSet pset;

    REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);

    pset.addParameter<int>("x", 1);
    psetDesc.validate(pset);

    psetDesc.addVPSet("y", edm::ParameterSetDescription());
    REQUIRE_THROWS_AS(psetDesc.validate(pset), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testWrongTrackiness") {
    edm::ParameterSet pset1;
    pset1.addParameter<int>("test1", 1);

    edm::ParameterSetDescription psetDesc1;
    psetDesc1.addUntracked<int>("test1", 1);
    REQUIRE_THROWS_AS(psetDesc1.validate(pset1), edm::Exception);

    edm::ParameterSet pset2;
    pset2.addParameter<edm::ParameterSet>("test2", edm::ParameterSet());

    edm::ParameterSetDescription psetDesc2;
    psetDesc2.addUntracked<edm::ParameterSetDescription>("test2", edm::ParameterSetDescription());
    REQUIRE_THROWS_AS(psetDesc2.validate(pset2), edm::Exception);

    edm::ParameterSet pset3;
    pset3.addParameter<std::vector<edm::ParameterSet>>("test3", std::vector<edm::ParameterSet>());

    edm::ParameterSetDescription psetDesc3;
    psetDesc3.addVPSetUntracked("test3", edm::ParameterSetDescription());
    REQUIRE_THROWS_AS(psetDesc3.validate(pset3), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testWrongType") {
    edm::ParameterSet pset1;
    pset1.addParameter<unsigned int>("test1", 1);

    edm::ParameterSetDescription psetDesc1;
    psetDesc1.add<int>("test1", 1);
    REQUIRE_THROWS_AS(psetDesc1.validate(pset1), edm::Exception);

    edm::ParameterSet pset2;
    pset2.addParameter<int>("test2", 1);

    edm::ParameterSetDescription psetDesc2;
    psetDesc2.add<edm::ParameterSetDescription>("test2", edm::ParameterSetDescription());
    REQUIRE_THROWS_AS(psetDesc2.validate(pset2), edm::Exception);

    edm::ParameterSet pset3;
    pset3.addParameter<int>("test3", 1);

    edm::ParameterSetDescription psetDesc3;
    psetDesc3.addVPSetUntracked("test3", edm::ParameterSetDescription());
    REQUIRE_THROWS_AS(psetDesc3.validate(pset3), edm::Exception);
  }

  // ---------------------------------------------------------------------------------

  SECTION("testPlugin") {
    static std::once_flag flag;
    std::call_once(flag, []() { edmplugin::PluginManager::configure(edmplugin::standard::config()); });
    {
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "ATestPlugin");
      pset1.addParameter<int>("anInt", 3);

      desc.validate(pset1);
    }

    {
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "BTestPlugin");
      pset1.addParameter<double>("aDouble", 0.2);

      desc.validate(pset1);
    }

    {
      //add defaults
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "ATestPlugin");
      desc.validate(pset1);
      CHECK(pset1.getParameter<int>("anInt") == 5);
    }

    {
      //add defaults
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", "ATestPlugin", true));

      edm::ParameterSet pset1;
      desc.validate(pset1);
      CHECK(pset1.getParameter<int>("anInt") == 5);
      CHECK(pset1.getParameter<std::string>("type") == "ATestPlugin");
    }

    {
      //an additional parameter
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "ATestPlugin");
      pset1.addParameter<int>("anInt", 3);
      pset1.addParameter<int>("NotRight", 3);

      REQUIRE_THROWS_AS(desc.validate(pset1), edm::Exception);
    }

    SECTION("wildcard") {
      //embedded with wildcard
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));
      edm::ParameterWildcard<edm::ParameterSetDescription> w("*", edm::RequireExactlyOne, true, desc);

      edm::ParameterSetDescription top;
      top.addNode(w);

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "ATestPlugin");

      edm::ParameterSet psetTop;
      psetTop.addParameter<edm::ParameterSet>("foo", pset1);

      top.validate(psetTop);
    }

    {
      //missing type
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<int>("anInt", 3);

      REQUIRE_THROWS_AS(desc.validate(pset1), edm::Exception);
    }

    {
      //a non-existent type
      edm::ParameterSetDescription desc;
      desc.addNode(edm::PluginDescription<TestPluginFactory>("type", true));

      edm::ParameterSet pset1;
      pset1.addParameter<std::string>("type", "ZTestPlugin");

      REQUIRE_THROWS_AS(desc.validate(pset1), cms::Exception);
    }
  }

  SECTION("VPSet with defaults") {
    edm::ParameterSetDescription psetDesc;

    std::vector<edm::ParameterSet> defaults{1};
    {
      auto& d = defaults.front();
      edm::ParameterSetDescription templte;

      templte.add<int>("i");
      d.addParameter<int>("i", 1);

      templte.add<std::vector<int>>("vi");
      d.addParameter<std::vector<int>>("vi", std::vector<int>({1}));

      templte.add<unsigned int>("ui");
      d.addParameter<unsigned int>("ui", 1);

      templte.add<std::vector<unsigned int>>("vui");
      d.addParameter<std::vector<unsigned int>>("vui", std::vector<unsigned int>({1}));

      templte.add<long long>("l");
      d.addParameter<long long>("l", 1);

      templte.add<std::vector<long long>>("vl");
      d.addParameter<std::vector<long long>>("vl", std::vector<long long>({1}));

      templte.add<unsigned long long>("ul");
      d.addParameter<unsigned long long>("ul", 1);

      templte.add<std::vector<unsigned long long>>("vul");
      d.addParameter<std::vector<unsigned long long>>("vul", std::vector<unsigned long long>({1}));

      templte.add<bool>("b");
      d.addParameter<bool>("b", true);

      templte.add<double>("d");
      d.addParameter<double>("d", 1.0);

      templte.add<std::vector<double>>("vd");
      d.addParameter<std::vector<double>>("vd", std::vector<double>({1.0}));

      templte.add<std::string>("s");
      d.addParameter<std::string>("s", "a");

      templte.add<std::vector<std::string>>("vs");
      d.addParameter<std::vector<std::string>>("vs", std::vector<std::string>({"a"}));

      templte.add<edm::InputTag>("t");
      d.addParameter<edm::InputTag>("t", edm::InputTag("foo"));

      templte.add<std::vector<edm::InputTag>>("vt");
      d.addParameter<std::vector<edm::InputTag>>("vt", std::vector<edm::InputTag>({edm::InputTag("foo")}));

      templte.add<edm::ESInputTag>("et");
      d.addParameter<edm::ESInputTag>("et", edm::ESInputTag(":foo"));

      templte.add<std::vector<edm::ESInputTag>>("vet");
      d.addParameter<std::vector<edm::ESInputTag>>("vet", std::vector<edm::ESInputTag>({edm::ESInputTag(":foo")}));

      edm::FileInPath::disableFileLookup();
      templte.add<edm::FileInPath>("f");
      d.addParameter<edm::FileInPath>("f", edm::FileInPath());

      templte.add<edm::EventID>("e");
      d.addParameter<edm::EventID>("e", edm::EventID(1, 2, 3));

      templte.add<std::vector<edm::EventID>>("ve");
      d.addParameter<std::vector<edm::EventID>>("ve", std::vector<edm::EventID>({edm::EventID(1, 2, 3)}));

      templte.add<edm::LuminosityBlockID>("L");
      d.addParameter<edm::LuminosityBlockID>("L", edm::LuminosityBlockID(1, 2));

      templte.add<std::vector<edm::LuminosityBlockID>>("vL");
      d.addParameter<std::vector<edm::LuminosityBlockID>>(
          "vL", std::vector<edm::LuminosityBlockID>({edm::LuminosityBlockID(1, 2)}));

      templte.add<edm::EventRange>("er");
      d.addParameter<edm::EventRange>("er", edm::EventRange(1, 2, 3, 4, 5, 6));

      templte.add<std::vector<edm::EventRange>>("ver");
      d.addParameter<std::vector<edm::EventRange>>("ver",
                                                   std::vector<edm::EventRange>({edm::EventRange(1, 2, 3, 4, 5, 6)}));

      templte.add<edm::LuminosityBlockRange>("Lr");
      d.addParameter<edm::LuminosityBlockRange>("Lr", edm::LuminosityBlockRange(1, 2, 3, 4));

      templte.add<std::vector<edm::LuminosityBlockRange>>("vLr");
      d.addParameter<std::vector<edm::LuminosityBlockRange>>(
          "vLr", std::vector<edm::LuminosityBlockRange>({edm::LuminosityBlockRange(1, 2, 3, 4)}));

      templte.add<edm::ParameterSetDescription>("p", edm::ParameterSetDescription());
      d.addParameter<edm::ParameterSet>("p", edm::ParameterSet());

      templte.addVPSet("vp", edm::ParameterSetDescription());
      d.addParameter<std::vector<edm::ParameterSet>>("vp", std::vector<edm::ParameterSet>());

      psetDesc.addVPSet("vp", templte, defaults);
    }
    SECTION("writeCfi full") {
      edm::ParameterSet test;
      test.addParameter("vp", defaults);
      psetDesc.validate(test);

      std::ostringstream s;
      edm::CfiOptions fullOps = edm::cfi::Typed{};
      psetDesc.writeCfi(s, false, 0, fullOps);
      std::string expected = R"-(
vp = cms.VPSet(
  cms.PSet(
    L = cms.LuminosityBlockID(1, 2),
    Lr = cms.LuminosityBlockRange('1:2-3:4'),
    b = cms.bool(True),
    d = cms.double(1),
    e = cms.EventID(1, 2, 3),
    er = cms.EventRange('1:2:3-4:5:6'),
    et = cms.ESInputTag('', 'foo'),
    f = cms.FileInPath(''),
    i = cms.int32(1),
    l = cms.int64(1),
    s = cms.string('a'),
    t = cms.InputTag('foo'),
    ui = cms.uint32(1),
    ul = cms.uint64(1),
    vL = cms.VLuminosityBlockID('1:2'),
    vLr = cms.VLuminosityBlockRange('1:2-3:4'),
    vd = cms.vdouble(1),
    ve = cms.VEventID('1:2:3'),
    ver = cms.VEventRange('1:2:3-4:5:6'),
    vet = cms.VESInputTag(':foo'),
    vi = cms.vint32(1),
    vl = cms.vint64(1),
    vs = cms.vstring('a'),
    vt = cms.VInputTag('foo'),
    vui = cms.vuint32(1),
    vul = cms.vuint64(1),
    p = cms.PSet(),
    vp = cms.VPSet(
    )
  ),
  template = cms.PSetTemplate(
    i = cms.required.int32,
    vi = cms.required.vint32,
    ui = cms.required.uint32,
    vui = cms.required.vuint32,
    l = cms.required.int64,
    vl = cms.required.vint64,
    ul = cms.required.uint64,
    vul = cms.required.vuint64,
    b = cms.required.bool,
    d = cms.required.double,
    vd = cms.required.vdouble,
    s = cms.required.string,
    vs = cms.required.vstring,
    t = cms.required.InputTag,
    vt = cms.required.VInputTag,
    et = cms.required.ESInputTag,
    vet = cms.required.VESInputTag,
    f = cms.required.FileInPath,
    e = cms.required.EventID,
    ve = cms.required.VEventID,
    L = cms.required.LuminosityBlockID,
    vL = cms.required.VLuminosityBlockID,
    er = cms.required.EventRange,
    ver = cms.required.VEventRange,
    Lr = cms.required.LuminosityBlockRange,
    vLr = cms.required.VLuminosityBlockRange,
    p = cms.PSet(),
    vp = cms.required.VPSet
  )
)
)-";

      CHECK(expected == s.str());
    }
    SECTION("writeCfi Untyped") {
      edm::ParameterSet test;
      test.addParameter("vp", defaults);
      psetDesc.validate(test);

      std::ostringstream s;
      edm::CfiOptions fullOps = edm::cfi::Untyped{edm::cfi::Paths{}};
      psetDesc.writeCfi(s, false, 0, fullOps);
      std::string expected = R"-(
vp = [
  cms.PSet(
    L = cms.LuminosityBlockID(1, 2),
    Lr = cms.LuminosityBlockRange('1:2-3:4'),
    b = cms.bool(True),
    d = cms.double(1),
    e = cms.EventID(1, 2, 3),
    er = cms.EventRange('1:2:3-4:5:6'),
    et = cms.ESInputTag('', 'foo'),
    f = cms.FileInPath(''),
    i = cms.int32(1),
    l = cms.int64(1),
    s = cms.string('a'),
    t = cms.InputTag('foo'),
    ui = cms.uint32(1),
    ul = cms.uint64(1),
    vL = cms.VLuminosityBlockID('1:2'),
    vLr = cms.VLuminosityBlockRange('1:2-3:4'),
    vd = cms.vdouble(1),
    ve = cms.VEventID('1:2:3'),
    ver = cms.VEventRange('1:2:3-4:5:6'),
    vet = cms.VESInputTag(':foo'),
    vi = cms.vint32(1),
    vl = cms.vint64(1),
    vs = cms.vstring('a'),
    vt = cms.VInputTag('foo'),
    vui = cms.vuint32(1),
    vul = cms.vuint64(1),
    p = cms.PSet(),
    vp = cms.VPSet(
    )
  )
]
)-";

      CHECK(expected == s.str());
    }
  }

  SECTION("PSet with default") {
    edm::ParameterSetDescription psetDesc;

    {
      edm::ParameterSetDescription templte;

      templte.add<int>("i", 1);
      templte.add<std::vector<int>>("vi", std::vector<int>({1}));
      templte.add<unsigned int>("ui", 1);
      templte.add<std::vector<unsigned int>>("vui", std::vector<unsigned int>({1}));
      templte.add<long long>("l", 1);
      templte.add<std::vector<long long>>("vl", std::vector<long long>({1}));
      templte.add<unsigned long long>("ul", 1);
      templte.add<std::vector<unsigned long long>>("vul", std::vector<unsigned long long>({1}));
      templte.add<bool>("b", true);
      templte.add<double>("d", 1.0);
      templte.add<std::vector<double>>("vd", std::vector<double>({1.0}));
      templte.add<std::string>("s", "a");
      templte.add<std::vector<std::string>>("vs", std::vector<std::string>({"a"}));
      templte.add<edm::InputTag>("t", edm::InputTag("foo"));
      templte.add<std::vector<edm::InputTag>>("vt", std::vector<edm::InputTag>({edm::InputTag("foo")}));
      templte.add<edm::ESInputTag>("et", edm::ESInputTag(":foo"));
      templte.add<std::vector<edm::ESInputTag>>("vet", std::vector<edm::ESInputTag>({edm::ESInputTag(":foo")}));
      edm::FileInPath::disableFileLookup();
      templte.add<edm::FileInPath>("f", edm::FileInPath());
      templte.add<edm::EventID>("e", edm::EventID(1, 2, 3));
      templte.add<std::vector<edm::EventID>>("ve", std::vector<edm::EventID>({edm::EventID(1, 2, 3)}));
      templte.add<edm::LuminosityBlockID>("L", edm::LuminosityBlockID(1, 2));
      templte.add<std::vector<edm::LuminosityBlockID>>(
          "vL", std::vector<edm::LuminosityBlockID>({edm::LuminosityBlockID(1, 2)}));
      templte.add<edm::EventRange>("er", edm::EventRange(1, 2, 3, 4, 5, 6));
      templte.add<std::vector<edm::EventRange>>("ver",
                                                std::vector<edm::EventRange>({edm::EventRange(1, 2, 3, 4, 5, 6)}));
      templte.add<edm::LuminosityBlockRange>("Lr", edm::LuminosityBlockRange(1, 2, 3, 4));
      templte.add<std::vector<edm::LuminosityBlockRange>>(
          "vLr", std::vector<edm::LuminosityBlockRange>({edm::LuminosityBlockRange(1, 2, 3, 4)}));
      templte.add<edm::ParameterSetDescription>("p", edm::ParameterSetDescription());
      templte.addVPSet("vp", edm::ParameterSetDescription(), std::vector<edm::ParameterSet>());
      psetDesc.add<edm::ParameterSetDescription>("p", templte);
    }
    SECTION("writeCfi full") {
      edm::ParameterSet test;
      test.addParameter("p", edm::ParameterSet());
      psetDesc.validate(test);

      std::ostringstream s;
      edm::CfiOptions fullOps = edm::cfi::Typed{};
      psetDesc.writeCfi(s, false, 0, fullOps);
      std::string expected = R"-(
p = cms.PSet(
  i = cms.int32(1),
  vi = cms.vint32(1),
  ui = cms.uint32(1),
  vui = cms.vuint32(1),
  l = cms.int64(1),
  vl = cms.vint64(1),
  ul = cms.uint64(1),
  vul = cms.vuint64(1),
  b = cms.bool(True),
  d = cms.double(1),
  vd = cms.vdouble(1),
  s = cms.string('a'),
  vs = cms.vstring('a'),
  t = cms.InputTag('foo'),
  vt = cms.VInputTag('foo'),
  et = cms.ESInputTag('', 'foo'),
  vet = cms.VESInputTag(':foo'),
  f = cms.FileInPath(''),
  e = cms.EventID(1, 2, 3),
  ve = cms.VEventID('1:2:3'),
  L = cms.LuminosityBlockID(1, 2),
  vL = cms.VLuminosityBlockID('1:2'),
  er = cms.EventRange('1:2:3-4:5:6'),
  ver = cms.VEventRange('1:2:3-4:5:6'),
  Lr = cms.LuminosityBlockRange('1:2-3:4'),
  vLr = cms.VLuminosityBlockRange('1:2-3:4'),
  p = cms.PSet(),
  vp = cms.VPSet(
  )
)
)-";

      CHECK(expected == s.str());
    }
    SECTION("writeCfi Untyped") {
      std::ostringstream s;
      edm::CfiOptions fullOps = edm::cfi::Untyped{edm::cfi::Paths{}};
      psetDesc.writeCfi(s, false, 0, fullOps);
      std::string expected = R"-(
p = dict(
  i = 1,
  vi = [1],
  ui = 1,
  vui = [1],
  l = 1,
  vl = [1],
  ul = 1,
  vul = [1],
  b = True,
  d = 1,
  vd = [1],
  s = 'a',
  vs = ['a'],
  t = ('foo'),
  vt = ['foo'],
  et = ('', 'foo'),
  vet = [':foo'],
  f = '',
  e = (1, 2, 3),
  ve = ['1:2:3'],
  L = (1, 2),
  vL = ['1:2'],
  er = ('1:2:3-4:5:6'),
  ver = ['1:2:3-4:5:6'],
  Lr = ('1:2-3:4'),
  vLr = ['1:2-3:4'],
  p = dict(),
  vp = [
  ]
)
)-";

      CHECK(expected == s.str());
    }
  }

  SECTION("setAllowAnything") {
    edm::ParameterSetDescription psetDesc;
    CHECK(!psetDesc.anythingAllowed());
    CHECK(!psetDesc.isUnknown());
    CHECK(psetDesc.begin() == psetDesc.end());

    edm::ParameterSet params;
    psetDesc.validate(params);

    params.addParameter<std::string>("testname", std::string("testvalue"));

    // Expect this to throw, parameter not in description
    REQUIRE_THROWS_AS(psetDesc.validate(params), edm::Exception);

    psetDesc.setAllowAnything();
    CHECK(psetDesc.anythingAllowed());

    psetDesc.validate(params);

    psetDesc.add<int>("testInt", 11);
    psetDesc.validate(params);
    CHECK(params.exists("testInt"));
  }

  SECTION("unknown") {
    edm::ParameterSetDescription psetDesc;

    edm::ParameterSet params;
    params.addParameter<std::string>("testname", std::string("testvalue"));
    psetDesc.setUnknown();
    CHECK(psetDesc.isUnknown());

    psetDesc.validate(params);
  }

  SECTION("FileInPath") {
    // Test this type separately because I do not know how to
    // add an entry into a ParameterSet without FileInPath pointing
    // at a real file.
    edm::ParameterSetDescription psetDesc;
    edm::ParameterDescriptionBase* par = psetDesc.add<edm::FileInPath>("fileInPath", edm::FileInPath());
    CHECK(par->type() == edm::k_FileInPath);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("FileInPath"));
  }

  SECTION("main") {
    edm::ParameterSetDescription psetDesc;
    edm::ParameterSet pset;

    psetDesc.reserve(2);

    int a = 1;
    edm::ParameterDescriptionBase* par = psetDesc.add<int>(std::string("ivalue"), a);
    CHECK(par->exists(pset) == false);
    CHECK(par->partiallyExists(pset) == false);
    CHECK(par->howManyXORSubNodesExist(pset) == 0);
    pset.addParameter<int>("ivalue", a);
    CHECK(par != 0);
    CHECK(par->label() == std::string("ivalue"));
    CHECK(par->type() == edm::k_int32);
    CHECK(par->isTracked() == true);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("int32"));
    CHECK(par->exists(pset) == true);
    CHECK(par->partiallyExists(pset) == true);
    CHECK(par->howManyXORSubNodesExist(pset) == 1);

    edm::ParameterSet psetWrongTrackiness;
    psetWrongTrackiness.addUntrackedParameter("ivalue", a);
    REQUIRE_THROWS_AS(psetDesc.validate(psetWrongTrackiness), edm::Exception);

    edm::ParameterSet psetWrongType;
    psetWrongType.addUntrackedParameter("ivalue", 1U);
    REQUIRE_THROWS_AS(psetDesc.validate(psetWrongType), edm::Exception);

    edm::ParameterSetDescription::const_iterator parIter = psetDesc.begin();
    CHECK(parIter->node().operator->() == par);

    unsigned b = 2;
    par = psetDesc.add<unsigned>("uvalue", b);
    pset.addParameter<unsigned>("uvalue", b);
    CHECK(par != 0);
    CHECK(par->label() == std::string("uvalue"));
    CHECK(par->type() == edm::k_uint32);
    CHECK(par->isTracked() == true);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("uint32"));

    parIter = psetDesc.begin();
    ++parIter;
    CHECK(parIter->node().operator->() == par);

    long long c = 3;
    par = psetDesc.addUntracked<long long>(std::string("i64value"), c);
    pset.addUntrackedParameter<long long>("i64value", c);
    CHECK(par != 0);
    CHECK(par->label() == std::string("i64value"));
    CHECK(par->type() == edm::k_int64);
    CHECK(par->isTracked() == false);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("int64"));

    unsigned long long d = 4;
    par = psetDesc.addUntracked<unsigned long long>("u64value", d);
    pset.addUntrackedParameter<unsigned long long>("u64value", d);
    CHECK(par != 0);
    CHECK(par->label() == std::string("u64value"));
    CHECK(par->type() == edm::k_uint64);
    CHECK(par->isTracked() == false);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("uint64"));

    double e = 5;
    par = psetDesc.addOptional<double>(std::string("dvalue"), e);
    pset.addParameter<double>("dvalue", e);
    CHECK(par != 0);
    CHECK(par->label() == std::string("dvalue"));
    CHECK(par->type() == edm::k_double);
    CHECK(par->isTracked() == true);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("double"));

    bool f = true;
    par = psetDesc.addOptional<bool>("bvalue", f);
    pset.addParameter<bool>("bvalue", f);
    CHECK(par != 0);
    CHECK(par->label() == std::string("bvalue"));
    CHECK(par->type() == edm::k_bool);
    CHECK(par->isTracked() == true);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("bool"));

    std::string g;
    par = psetDesc.addOptionalUntracked<std::string>(std::string("svalue"), g);
    pset.addUntrackedParameter<std::string>("svalue", g);
    CHECK(par != 0);
    CHECK(par->label() == std::string("svalue"));
    CHECK(par->type() == edm::k_stringRaw);
    CHECK(par->isTracked() == false);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("string"));

    edm::EventID h;
    par = psetDesc.addOptionalUntracked<edm::EventID>("evalue", h);
    pset.addUntrackedParameter<edm::EventID>("evalue", h);
    CHECK(par != 0);
    CHECK(par->label() == std::string("evalue"));
    CHECK(par->type() == edm::k_EventID);
    CHECK(par->isTracked() == false);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("EventID"));

    edm::LuminosityBlockID i;
    par = psetDesc.add<edm::LuminosityBlockID>("lvalue", i);
    pset.addParameter<edm::LuminosityBlockID>("lvalue", i);
    CHECK(par->type() == edm::k_LuminosityBlockID);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("LuminosityBlockID"));

    edm::InputTag j;
    par = psetDesc.add<edm::InputTag>("input", j);
    pset.addParameter<edm::InputTag>("input", j);
    CHECK(par->type() == edm::k_InputTag);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("InputTag"));

    edm::ESInputTag k;
    par = psetDesc.add<edm::ESInputTag>("esinput", k);
    pset.addParameter<edm::ESInputTag>("esinput", k);
    CHECK(par->type() == edm::k_ESInputTag);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("ESInputTag"));

    std::vector<int> v1;
    par = psetDesc.add<std::vector<int>>("v1", v1);
    pset.addParameter<std::vector<int>>("v1", v1);
    CHECK(par->type() == edm::k_vint32);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vint32"));

    std::vector<unsigned> v2;
    par = psetDesc.add<std::vector<unsigned>>("v2", v2);
    pset.addParameter<std::vector<unsigned>>("v2", v2);
    CHECK(par->type() == edm::k_vuint32);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vuint32"));

    std::vector<long long> v3;
    par = psetDesc.add<std::vector<long long>>("v3", v3);
    pset.addParameter<std::vector<long long>>("v3", v3);
    CHECK(par->type() == edm::k_vint64);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vint64"));

    std::vector<unsigned long long> v4;
    par = psetDesc.add<std::vector<unsigned long long>>("v4", v4);
    pset.addParameter<std::vector<unsigned long long>>("v4", v4);
    CHECK(par->type() == edm::k_vuint64);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vuint64"));

    std::vector<double> v5;
    par = psetDesc.add<std::vector<double>>("v5", v5);
    pset.addParameter<std::vector<double>>("v5", v5);
    CHECK(par->type() == edm::k_vdouble);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vdouble"));

    std::vector<std::string> v6;
    par = psetDesc.add<std::vector<std::string>>("v6", v6);
    pset.addParameter<std::vector<std::string>>("v6", v6);
    CHECK(par->type() == edm::k_vstringRaw);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("vstring"));

    std::vector<edm::EventID> v7;
    par = psetDesc.add<std::vector<edm::EventID>>("v7", v7);
    pset.addParameter<std::vector<edm::EventID>>("v7", v7);
    CHECK(par->type() == edm::k_VEventID);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("VEventID"));

    std::vector<edm::LuminosityBlockID> v8;
    par = psetDesc.add<std::vector<edm::LuminosityBlockID>>("v8", v8);
    pset.addParameter<std::vector<edm::LuminosityBlockID>>("v8", v8);
    CHECK(par->type() == edm::k_VLuminosityBlockID);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("VLuminosityBlockID"));

    std::vector<edm::InputTag> v9;
    par = psetDesc.add<std::vector<edm::InputTag>>("v9", v9);
    pset.addParameter<std::vector<edm::InputTag>>("v9", v9);
    CHECK(par->type() == edm::k_VInputTag);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("VInputTag"));

    std::vector<edm::ESInputTag> v11;
    par = psetDesc.add<std::vector<edm::ESInputTag>>("v11", v11);
    pset.addParameter<std::vector<edm::ESInputTag>>("v11", v11);
    CHECK(par->type() == edm::k_VESInputTag);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("VESInputTag"));

    edm::ParameterSetDescription m;
    par = psetDesc.add<edm::ParameterSetDescription>("psetDesc", m);
    CHECK(par->exists(pset) == false);
    CHECK(par->partiallyExists(pset) == false);
    CHECK(par->howManyXORSubNodesExist(pset) == 0);
    edm::ParameterSet p1;
    pset.addParameter<edm::ParameterSet>("psetDesc", p1);
    CHECK(par->type() == edm::k_PSet);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("PSet"));
    CHECK(par->exists(pset) == true);
    CHECK(par->partiallyExists(pset) == true);
    CHECK(par->howManyXORSubNodesExist(pset) == 1);

    edm::ParameterSetDescription v10;
    par = psetDesc.addVPSet("psetVectorDesc", v10);
    CHECK(par->exists(pset) == false);
    CHECK(par->partiallyExists(pset) == false);
    CHECK(par->howManyXORSubNodesExist(pset) == 0);
    std::vector<edm::ParameterSet> vp1;
    pset.addParameter<std::vector<edm::ParameterSet>>("psetVectorDesc", vp1);
    CHECK(par->type() == edm::k_VPSet);
    CHECK(edm::parameterTypeEnumToString(par->type()) == std::string("VPSet"));
    CHECK(par->exists(pset) == true);
    CHECK(par->partiallyExists(pset) == true);
    CHECK(par->howManyXORSubNodesExist(pset) == 1);

    psetDesc.validate(pset);

    // Add a ParameterSetDescription nested in a ParameterSetDescription nested in
    // a vector in the top level ParameterSetDescription to see if the nesting is
    // working properly.

    edm::ParameterSet nest2;
    nest2.addParameter<int>("intLevel2a", 1);
    nest2.addUntrackedParameter<int>("intLevel2b", 1);
    nest2.addParameter<int>("intLevel2e", 1);
    nest2.addUntrackedParameter<int>("intLevel2f", 1);

    edm::ParameterSet nest1;
    nest1.addParameter<int>("intLevel1a", 1);
    nest1.addParameter<edm::ParameterSet>("nestLevel1b", nest2);

    std::vector<edm::ParameterSet> vPset;
    vPset.push_back(edm::ParameterSet());
    vPset.push_back(nest1);

    pset.addUntrackedParameter<std::vector<edm::ParameterSet>>("nestLevel0", vPset);

    std::vector<edm::ParameterSetDescription> testDescriptions;
    testDescriptions.push_back(psetDesc);
    testDescriptions.push_back(psetDesc);
    testDescriptions.push_back(psetDesc);

    for (int ii = 0; ii < 3; ++ii) {
      edm::ParameterSetDescription nestLevel2;

      // for the first test do not put a parameter in the description
      // so there will be an extra parameter in the ParameterSet and
      // validation should fail.
      if (ii > 0)
        nestLevel2.add<int>("intLevel2a", 1);

      // for the next test validation should pass

      // For the last test add an extra required parameter in the
      // description that is not in the ParameterSet.
      if (ii == 2)
        nestLevel2.add<int>("intLevel2extra", 11);

      nestLevel2.addUntracked<int>("intLevel2b", 1);
      nestLevel2.addOptional<int>("intLevel2c", 1);
      nestLevel2.addOptionalUntracked<int>("intLevel2d", 1);
      nestLevel2.addOptional<int>("intLevel2e", 1);
      nestLevel2.addOptionalUntracked<int>("intLevel2f", 1);

      edm::ParameterSetDescription nestLevel1;
      nestLevel1.add<int>("intLevel1a", 1);
      nestLevel1.add<edm::ParameterSetDescription>("nestLevel1b", nestLevel2);

      testDescriptions[ii].addVPSetUntracked("nestLevel0", nestLevel1);
    }

    // Now run the validation and make sure we get the expected results
    REQUIRE_THROWS_AS(testDescriptions[0].validate(pset), edm::Exception);

    // This one should pass validation with no exception
    testDescriptions[1].validate(pset);

    // This validation should also pass and it should insert
    // the missing parameter into the ParameterSet
    testDescriptions[2].validate(pset);

    std::vector<edm::ParameterSet> const& vpset = pset.getUntrackedParameterSetVector("nestLevel0");
    edm::ParameterSet const& psetInPset = vpset[1].getParameterSet("nestLevel1b");
    CHECK(psetInPset.getParameter<int>("intLevel2extra") == 11);

    // One more iteration, this time the purpose is to
    // test the parameterSetDescription accessors.
    edm::ParameterSetDescription nestLevel2;
    par = nestLevel2.add<int>("intLevel2a", 1);
    par->setComment("testComment");
    CHECK(par->parameterSetDescription() == 0);
    edm::ParameterDescriptionBase const& constParRef = *par;
    CHECK(constParRef.parameterSetDescription() == 0);

    nestLevel2.addUntracked<int>("intLevel2b", 1);
    nestLevel2.addOptional<int>("intLevel2c", 1);
    nestLevel2.addOptionalUntracked<int>("intLevel2d", 1);
    nestLevel2.addOptional<int>("intLevel2e", 1);
    nestLevel2.addOptionalUntracked<int>("intLevel2f", 1);
    nestLevel2.setAllowAnything();

    edm::ParameterSetDescription nestLevel1;
    par = nestLevel1.add<int>("intLevel1a", 1);
    par->setComment("testComment1");
    par = nestLevel1.add<edm::ParameterSetDescription>("nestLevel1b", nestLevel2);
    CHECK(par->parameterSetDescription() != 0);
    CHECK(par->parameterSetDescription()->begin()->node()->comment() == std::string("testComment"));
    edm::ParameterDescriptionBase const& constParRef2 = *par;
    CHECK(constParRef2.parameterSetDescription() != 0);
    CHECK(constParRef2.parameterSetDescription()->begin()->node()->comment() == std::string("testComment"));

    CHECK(par->parameterSetDescription()->anythingAllowed() == true);
    CHECK(constParRef2.parameterSetDescription()->anythingAllowed() == true);

    par = psetDesc.addVPSetUntracked("nestLevel0", nestLevel1);
    CHECK(par->parameterSetDescription() != 0);
    CHECK(par->parameterSetDescription()->begin()->node()->comment() == std::string("testComment1"));
    edm::ParameterDescriptionBase const& constParRef3 = *par;
    CHECK(constParRef3.parameterSetDescription() != 0);
    CHECK(constParRef3.parameterSetDescription()->begin()->node()->comment() == std::string("testComment1"));

    psetDesc.validate(pset);
  }
}  // namespace testParameterSetDescription

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TestPluginFactory, "TestPluginFWCoreParameterSet");

DEFINE_EDM_VALIDATED_PLUGIN(TestPluginFactory, testParameterSetDescription::ATestPlugin, "ATestPlugin");
DEFINE_EDM_VALIDATED_PLUGIN(TestPluginFactory, testParameterSetDescription::BTestPlugin, "BTestPlugin");
