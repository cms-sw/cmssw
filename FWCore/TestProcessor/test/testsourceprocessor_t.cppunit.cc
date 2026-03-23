#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestSourceProcessor.h"

#include <cppunit/extensions/HelperMacros.h>

class testTestSourceProcessor : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTestSourceProcessor);
  CPPUNIT_TEST(simpleTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void simpleTest();

private:
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTestSourceProcessor);

void testTestSourceProcessor::simpleTest() {
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
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsFile);
    auto f = tester.openFile();
    CPPUNIT_ASSERT(bool(f));
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsRun);
    auto r = tester.readRun();
    CPPUNIT_ASSERT(r.run() == 1);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsLumi);
    auto r = tester.readLuminosityBlock();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsEvent);
    auto r = tester.readEvent();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
    CPPUNIT_ASSERT(r.event() == 1);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsEvent);
    auto r = tester.readEvent();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
    CPPUNIT_ASSERT(r.event() == 2);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsEvent);
    auto r = tester.readEvent();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
    CPPUNIT_ASSERT(r.event() == 3);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsEvent);
    auto r = tester.readEvent();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
    CPPUNIT_ASSERT(r.event() == 4);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsEvent);
    auto r = tester.readEvent();
    CPPUNIT_ASSERT(r.run() == 1);
    CPPUNIT_ASSERT(r.luminosityBlock() == 1);
    CPPUNIT_ASSERT(r.event() == 5);
  }
  {
    auto n = tester.findNextTransition();
    CPPUNIT_ASSERT(n == edm::InputSource::ItemType::IsStop);
  }
}
