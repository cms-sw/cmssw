/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT.

 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <vector>
#include "catch2/catch_all.hpp"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "TFile.h"
#include "TSystem.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"

#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/MultiChainEvent.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

class testRefInROOT {
public:
  testRefInROOT() {}
  void setUp() {
    if (!sWasRun_) {
      FWLiteEnabler::enable();
      sWasRun_ = true;
    }
    tmpdir = "./";
  }
  void tearDown() {}

  void testRefFirst();
  void testAllLabels();
  void testOneGoodFile();
  void testTwoGoodFiles();
  void failOneBadFile();
  void testGoodChain();
  void testHandleErrors();
  void testMissingRef();
  void testMissingData();
  void testEventBase();
  void testSometimesMissingData();
  void testTo();
  void failChainWithMissingFile();
  //void failDidNotCallGetEntryForEvents();

private:
  static bool sWasRun_;
  std::string tmpdir;
};

bool testRefInROOT::sWasRun_ = false;

static void checkMatch(const edmtest::OtherThingCollection* pOthers, const edmtest::ThingCollection* pThings) {
  REQUIRE(pOthers != nullptr);
  REQUIRE(pThings != nullptr);
  REQUIRE(pOthers->size() == pThings->size());

  //This test requires at least one entry
  REQUIRE(pOthers->size() > 0);
  const edm::View<edmtest::Thing>& view = *(pOthers->front().refToBaseProd);
  REQUIRE(view.size() == pOthers->size());

  edmtest::ThingCollection::const_iterator itThing = pThings->begin(), itThingEnd = pThings->end();
  edmtest::OtherThingCollection::const_iterator itOther = pOthers->begin();
  edm::View<edmtest::Thing>::const_iterator itView = view.begin();

  for (; itThing != itThingEnd; ++itThing, ++itOther, ++itView) {
    //std::cout <<"getting data"<<std::endl;
    //I'm assuming the following is true
    REQUIRE(itOther->ref.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    REQUIRE(itOther->ref.get()->a == itThing->a);
    if (itView->a != itThing->a) {
      std::cout << " *PROBLEM: RefToBaseProd " << itView->a << "!= thing " << itThing->a << std::endl;
    }
    REQUIRE(itView->a == itThing->a);
  }
}

static void testEvent(fwlite::Event& events) {
  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::ThingCollection> pThings;
    pThings.getByLabel(events, "Thing");

    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");

    checkMatch(pOthers.ptr(), pThings.ptr());
  }
}

void testRefInROOT::testOneGoodFile() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  testEvent(events);
}

void testRefInROOT::testAllLabels() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag", "TEST");
  }
}

void testRefInROOT::testEventBase() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);
  edm::InputTag tagFull("OtherThing", "testUserTag", "TEST");
  edm::InputTag tag("OtherThing", "testUserTag");
  edm::InputTag tagNotHere("NotHereOtherThing");
  edm::InputTag tagThing("Thing");
  edm::EventBase* eventBase = &events;

  for (events.toBegin(); not events.atEnd(); ++events) {
    {
      edm::Handle<edmtest::OtherThingCollection> pOthers;
      eventBase->getByLabel(tagFull, pOthers);
      REQUIRE(pOthers.isValid());

      // Test that the get function that takes a ProductID works
      // by getting a ProductID from a Ref stored in the OtherThingCollection
      // and testing that one can retrieve the ThingCollection with it.
      REQUIRE(pOthers->size() > 0);
      edmtest::OtherThingCollection::const_iterator itOther = pOthers->begin();
      edm::ProductID thingProductID = itOther->ref.id();
      edm::Handle<edmtest::ThingCollection> thingCollectionHandle;
      eventBase->get(thingProductID, thingCollectionHandle);
      edm::Handle<edmtest::ThingCollection> thingCollectionHandle2;
      eventBase->getByLabel(tagThing, thingCollectionHandle2);
      REQUIRE(thingCollectionHandle.product() == thingCollectionHandle2.product());
      REQUIRE(thingCollectionHandle.product()->begin()->a == thingCollectionHandle2.product()->begin()->a);
    }
    {
      edm::Handle<edmtest::OtherThingCollection> pOthers;
      eventBase->getByLabel(tag, pOthers);
      REQUIRE(pOthers.isValid());
      REQUIRE(pOthers->size() > 0);
    }

    {
      edm::Handle<edmtest::OtherThingCollection> pOthers;
      eventBase->getByLabel(tagNotHere, pOthers);

      REQUIRE(not pOthers.isValid());
      REQUIRE(pOthers.failedToGet());
      REQUIRE_THROWS_AS(pOthers.product(), cms::Exception);
    }
  }
}

void testRefInROOT::testTo() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);
  edm::InputTag tag("Thing");
  edm::EventBase* eventBase = &events;

  REQUIRE(events.to(1, 1, 2));
  {
    edm::Handle<edmtest::ThingCollection> pThings;
    eventBase->getByLabel(tag, pThings);
    REQUIRE(pThings.isValid());
    REQUIRE(0 != pThings->size());
    REQUIRE(3 == (*pThings)[0].a);
  }
  std::cout << events.id() << std::endl;
  REQUIRE(edm::EventID(1, 1, 2) == events.id());

  REQUIRE(events.to(1, 1, 1));
  {
    edm::Handle<edmtest::ThingCollection> pThings;
    eventBase->getByLabel(tag, pThings);
    REQUIRE(pThings.isValid());
    REQUIRE(0 != pThings->size());
    REQUIRE(2 == (*pThings)[0].a);
  }
  REQUIRE(edm::EventID(1, 1, 1) == events.id());

  REQUIRE(events.to(1));
  REQUIRE(not events.to(events.size()));
}

void testRefInROOT::testRefFirst() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");

    //std::cout <<"got OtherThing"<<std::endl;
    for (auto const& other : *pOthers) {
      //std::cout <<"getting ref"<<std::endl;
      int arbitraryBigNumber = 1000000;
      REQUIRE(other.ref.get()->a < arbitraryBigNumber);
    }
    //std::cout <<"get all Refs"<<std::endl;

    fwlite::Handle<edmtest::ThingCollection> pThings;
    pThings.getByLabel(events, "Thing");

    //std::cout <<"checkMatch"<<std::endl;
    checkMatch(pOthers.ptr(), pThings.ptr());
  }
}

void testRefInROOT::failOneBadFile() {
  TFile file("thisFileDoesNotExist.root");
  fwlite::Event events(&file);

  testEvent(events);
}

void testRefInROOT::testMissingRef() {
  TFile file((tmpdir + "other_onlyDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");
    for (auto const& other : *pOthers) {
      //std::cout <<"getting ref"<<std::endl;
      REQUIRE(not other.ref.isAvailable());
      REQUIRE_THROWS_AS(other.ref.get(), cms::Exception);
    }
  }
}

void testRefInROOT::testMissingData() {
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "NotHereOtherThing");

    REQUIRE(not pOthers.isValid());
    REQUIRE(pOthers.failedToGet());
    REQUIRE_THROWS_AS(pOthers.product(), cms::Exception);
  }
}

void testRefInROOT::testSometimesMissingData() {
  TFile file((tmpdir + "partialEventDataFormatsFWLite.root").c_str());
  fwlite::Event events(&file);

  unsigned int index = 0;
  edm::InputTag tag("OtherThing", "testUserTag");
  for (events.toBegin(); not events.atEnd(); ++events, ++index) {
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");

    if (0 == index) {
      REQUIRE(not pOthers.isValid());
      REQUIRE(pOthers.failedToGet());
      REQUIRE_THROWS_AS(pOthers.product(), cms::Exception);
    } else {
      REQUIRE(pOthers.isValid());
    }

    edm::Handle<edmtest::OtherThingCollection> edmPOthers;
    events.getByLabel(tag, edmPOthers);
    if (0 == index) {
      REQUIRE(not edmPOthers.isValid());
      REQUIRE(edmPOthers.failedToGet());
      REQUIRE_THROWS_AS(edmPOthers.product(), cms::Exception);
    } else {
      REQUIRE(edmPOthers.isValid());
    }
  }
}

void testRefInROOT::testHandleErrors() {
  fwlite::Handle<edmtest::ThingCollection> pThings;
  REQUIRE_THROWS_AS(*pThings, cms::Exception);

  //try copy constructor
  fwlite::Handle<edmtest::ThingCollection> pThings2(pThings);
  REQUIRE_THROWS_AS(*pThings2, cms::Exception);
}

void testRefInROOT::testTwoGoodFiles() {
  /*
  std::cout <<"gFile "<<gFile<<std::endl;
  TFile file((tmpdir + "goodDataFormatsFWLite.root").c_str());
  std::cout <<" file :" << &file<<" gFile: "<<gFile<<std::endl;
  
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName()));
  
  testTree(events);
  std::cout <<"working on second file"<<std::endl;
  TFile file2((tmpdir + "good2DataFormatsFWLite.root").c_str());
  std::cout <<" file2 :"<< &file2<<" gFile: "<<gFile<<std::endl;
  events = dynamic_cast<TTree*>(file2.Get(edm::poolNames::eventTreeName()));
  
  testTree(events);
   */
}

void testRefInROOT::testGoodChain() {
  std::vector<std::string> files{tmpdir + "goodDataFormatsFWLite.root", tmpdir + "good2DataFormatsFWLite.root"};
  fwlite::ChainEvent events(files);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::ThingCollection> pThings;
    pThings.getByLabel(events, "Thing");

    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");

    checkMatch(pOthers.ptr(), pThings.ptr());
  }
}

void testRefInROOT::failChainWithMissingFile() {
  std::vector<std::string> files{tmpdir + "goodDataFormatsFWLite.root", tmpdir + "2ndFileDoesNotExist.root"};
  fwlite::ChainEvent events(files);

  for (events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::ThingCollection> pThings;
    pThings.getByLabel(events, "Thing");

    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events, "OtherThing", "testUserTag");

    checkMatch(pOthers.ptr(), pThings.ptr());
  }
}

TEST_CASE("FWLite Ref tests", "[FWLite]") {
  testRefInROOT test;
  test.setUp();

  SECTION("testOneGoodFile") { test.testOneGoodFile(); }
  SECTION("failOneBadFile") { REQUIRE_THROWS_AS(test.failOneBadFile(), std::exception); }
  SECTION("failChainWithMissingFile") { REQUIRE_THROWS_AS(test.failChainWithMissingFile(), std::exception); }
  SECTION("testRefFirst") { test.testRefFirst(); }
  SECTION("testAllLabels") { test.testAllLabels(); }
  SECTION("testGoodChain") { test.testGoodChain(); }
  SECTION("testTwoGoodFiles") { test.testTwoGoodFiles(); }
  SECTION("testHandleErrors") { test.testHandleErrors(); }
  SECTION("testMissingRef") { test.testMissingRef(); }
  SECTION("testMissingData") { test.testMissingData(); }
  SECTION("testEventBase") { test.testEventBase(); }
  SECTION("testSometimesMissingData") { test.testSometimesMissingData(); }
  SECTION("testTo") { test.testTo(); }
}
