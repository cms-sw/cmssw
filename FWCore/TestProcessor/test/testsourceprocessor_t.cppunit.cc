#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestSourceProcessor.h"

#include "catch2/catch_all.hpp"

TEST_CASE("TestSourceProcessor tests", "[TestSourceProcessor]") {
  SECTION("simple test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestSourceProcess import *\n"
        "process = TestSourceProcess()\n"
        "process.source = cms.Source('TestSource',"
        "transitions = cms.untracked.VPSet(\n"
        "cms.PSet(type = cms.untracked.string('IsFile'),\n"
        "         id = cms.untracked.EventID(0,0,0)),\n"
        "cms.PSet(type = cms.untracked.string('IsRun'),\n"
        "         id = cms.untracked.EventID(1,0,0)),\n"
        "cms.PSet(type = cms.untracked.string('IsLumi'),\n"
        "         id = cms.untracked.EventID(1,1,0)),\n"
        "cms.PSet(type = cms.untracked.string('IsEvent'),\n"
        "         id = cms.untracked.EventID(1,1,1)),\n"
        "cms.PSet(type = cms.untracked.string('IsEvent'),\n"
        "         id = cms.untracked.EventID(1,1,2)),\n"
        "cms.PSet(type = cms.untracked.string('IsEvent'),\n"
        "         id = cms.untracked.EventID(1,1,3)),\n"
        "cms.PSet(type = cms.untracked.string('IsEvent'),\n"
        "         id = cms.untracked.EventID(1,1,4)),\n"
        "cms.PSet(type = cms.untracked.string('IsEvent'),\n"
        "         id = cms.untracked.EventID(1,1,5)),\n"
        "cms.PSet(type = cms.untracked.string('IsStop'),\n"
        "         id = cms.untracked.EventID(0,0,0))\n"
        "))\n";
    edm::test::TestSourceProcessor tester(kTest);

    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsFile);
      auto f = tester.openFile();
      REQUIRE(bool(f));
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsRun);
      auto r = tester.readRun();
      REQUIRE(r.run() == 1);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsLumi);
      auto r = tester.readLuminosityBlock();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsEvent);
      auto r = tester.readEvent();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
      REQUIRE(r.event() == 1);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsEvent);
      auto r = tester.readEvent();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
      REQUIRE(r.event() == 2);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsEvent);
      auto r = tester.readEvent();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
      REQUIRE(r.event() == 3);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsEvent);
      auto r = tester.readEvent();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
      REQUIRE(r.event() == 4);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsEvent);
      auto r = tester.readEvent();
      REQUIRE(r.run() == 1);
      REQUIRE(r.luminosityBlock() == 1);
      REQUIRE(r.event() == 5);
    }
    {
      auto n = tester.findNextTransition();
      REQUIRE(n == edm::InputSource::ItemType::IsStop);
    }
  }
}
