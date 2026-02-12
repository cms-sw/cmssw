#include "FWCore/ParameterSet/interface/DescriptionCloner.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "catch2/catch_all.hpp"
#include <string>

TEST_CASE("DescriptionCloner") {
  SECTION("basic usage") {
    edm::DescriptionCloner cloner;
    cloner.set("level1.level2.intParam", 42);
    cloner.set("level1.level2.stringParam", std::string("test"));
    cloner.set("boolParam", true);

    SECTION("default tracked no value") {
      edm::ParameterSetDescription defaultDesc;
      edm::ParameterSetDescription level2Desc;
      level2Desc.add<int>("intParam");
      level2Desc.add<std::string>("stringParam");
      edm::ParameterSetDescription level1Desc;
      level1Desc.add<edm::ParameterSetDescription>("level2", level2Desc);
      defaultDesc.add<edm::ParameterSetDescription>("level1", level1Desc);
      defaultDesc.add<bool>("boolParam");

      cloner.determineTrackinessFromDefaultDescription(defaultDesc);

      auto diffDesc = cloner.createDifference();

      edm::ParameterSet ps;
      diffDesc.validate(ps);

      auto const& l1 = ps.getParameter<edm::ParameterSet>("level1");
      auto const& l2 = l1.getParameter<edm::ParameterSet>("level2");
      REQUIRE(l2.getParameter<int>("intParam") == 42);
      REQUIRE(l2.getParameter<std::string>("stringParam") == "test");
      REQUIRE(ps.getParameter<bool>("boolParam") == true);
    }
    SECTION("default untracked no value") {
      edm::ParameterSetDescription defaultDesc;
      edm::ParameterSetDescription level2Desc;
      level2Desc.addUntracked<int>("intParam");
      level2Desc.addUntracked<std::string>("stringParam");
      edm::ParameterSetDescription level1Desc;
      level1Desc.addUntracked<edm::ParameterSetDescription>("level2", level2Desc);
      defaultDesc.addUntracked<edm::ParameterSetDescription>("level1", level1Desc);
      defaultDesc.addUntracked<bool>("boolParam");

      cloner.determineTrackinessFromDefaultDescription(defaultDesc);

      auto diffDesc = cloner.createDifference();

      edm::ParameterSet ps;
      diffDesc.validate(ps);

      auto const& l1 = ps.getUntrackedParameterSet("level1");
      auto const& l2 = l1.getUntrackedParameterSet("level2");
      REQUIRE(l2.getUntrackedParameter<int>("intParam") == 42);
      REQUIRE(l2.getUntrackedParameter<std::string>("stringParam") == "test");
      REQUIRE(ps.getUntrackedParameter<bool>("boolParam") == true);
    }
    SECTION("default with value") {
      edm::ParameterSetDescription defaultDesc;
      edm::ParameterSetDescription level2Desc;
      level2Desc.add<int>("intParam", 100);
      level2Desc.add<std::string>("stringParam", "default");
      edm::ParameterSetDescription level1Desc;
      level1Desc.add<edm::ParameterSetDescription>("level2", level2Desc);
      defaultDesc.add<edm::ParameterSetDescription>("level1", level1Desc);
      defaultDesc.add<bool>("boolParam", false);

      cloner.determineTrackinessFromDefaultDescription(defaultDesc);

      auto diffDesc = cloner.createDifference();

      edm::ParameterSet ps;
      diffDesc.validate(ps);

      auto const& l1 = ps.getParameter<edm::ParameterSet>("level1");
      auto const& l2 = l1.getParameter<edm::ParameterSet>("level2");
      REQUIRE(l2.getParameter<int>("intParam") == 42);
      REQUIRE(l2.getParameter<std::string>("stringParam") == "test");
      REQUIRE(ps.getParameter<bool>("boolParam") == true);
    }
  }
  SECTION("empty cloner") {
    edm::DescriptionCloner cloner;

    edm::ParameterSetDescription defaultDesc;
    defaultDesc.add<int>("someParam", 10);

    cloner.determineTrackinessFromDefaultDescription(defaultDesc);

    auto diffDesc = cloner.createDifference();

    edm::ParameterSet ps;
    diffDesc.validate(ps);

    REQUIRE(ps.exists("someParam") == false);
  }
  SECTION("invalid path") {
    edm::DescriptionCloner cloner;
    cloner.set("validParam", 1);

    edm::ParameterSetDescription defaultDesc;
    defaultDesc.add<int>("validParam", 0);
    cloner.set("invalid.level.param", 2);

    REQUIRE_THROWS_AS(cloner.determineTrackinessFromDefaultDescription(defaultDesc), edm::Exception);
  }
  SECTION("explicit trackiness") {
    edm::DescriptionCloner cloner;
    cloner.set("level1.+level2.-intParam", 42);
    cloner.set("level1.+level2.+stringParam", std::string("test"));
    cloner.set("boolParam", true);

    edm::ParameterSetDescription defaultDesc;
    defaultDesc.addUntracked<bool>("boolParam");
    edm::ParameterSetDescription level1Desc;
    level1Desc.setAllowAnything();
    defaultDesc.add("level1", level1Desc);

    edm::ParameterSet ps;
    cloner.determineTrackinessFromDefaultDescription(defaultDesc);
    const auto diffDesc = cloner.createDifference();
    diffDesc.validate(ps);
    auto const& l1 = ps.getParameter<edm::ParameterSet>("level1");
    auto const& l2 = l1.getParameter<edm::ParameterSet>("level2");
    REQUIRE(l2.getUntrackedParameter<int>("intParam") == 42);
    REQUIRE(l2.getParameter<std::string>("stringParam") == "test");
    REQUIRE(ps.getUntrackedParameter<bool>("boolParam") == true);
  }
  SECTION("incorrect explicit trackiness") {
    edm::DescriptionCloner cloner;
    cloner.set("level1.-level2.-intParam", 42);
    cloner.set("level1.-level2.-stringParam", std::string("test"));
    cloner.set("boolParam", true);

    edm::ParameterSetDescription defaultDesc;
    defaultDesc.addUntracked<bool>("boolParam");
    edm::ParameterSetDescription level1Desc;
    level1Desc.setAllowAnything();
    defaultDesc.add("level1", level1Desc);

    edm::ParameterSet ps;

    cloner.determineTrackinessFromDefaultDescription(defaultDesc);
    const auto diffDesc = cloner.createDifference();
    diffDesc.validate(ps);
    auto const& l1 = ps.getParameter<edm::ParameterSet>("level1");
    REQUIRE_THROWS_AS(l1.getParameter<edm::ParameterSet>("level2"), edm::Exception);
  }
  SECTION("inconsistent explicit trackiness") {
    edm::DescriptionCloner cloner;
    cloner.set("level1.-level2.-intParam", 42);
    cloner.set("level1.+level2.-stringParam", std::string("test"));
    cloner.set("boolParam", true);

    edm::ParameterSetDescription defaultDesc;
    defaultDesc.addUntracked<bool>("boolParam");
    edm::ParameterSetDescription level1Desc;
    level1Desc.setAllowAnything();
    defaultDesc.add("level1", level1Desc);

    edm::ParameterSet ps;

    cloner.determineTrackinessFromDefaultDescription(defaultDesc);

    REQUIRE_THROWS_AS(cloner.createDifference(), edm::Exception);
  }
}
