/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT using the
BareRootProductGetter.

One of the main purposes of this package is to run ROOT
directly and be able to use Refs. In addition to this
automated test, I ran the following manually to verify
this was working:

This runs the FWLite Ref testing from FWCore/FWLite
(you might need to edit the SCRAM architecture)

../tmp/slc6_amd64_gcc481/src/FWCore/FWLite/test/testFWCoreFWLite/testFWCoreFWLite

It will produce a file called good.root. If you want
to run Draw using the BareRootProductGetter, then use
the following sequence of commands.

  cmsenv
  root.exe
  gSystem->Load("libFWCoreFWLite.so");
  FWLiteEnabler::enable();
  TFile f("good.root");

Then this one will draw just a simple variable:

  Events.Draw("edmtestThings_Thing__TEST.obj.a")

This runs Draw through a Ref using the BareRootProductGetter:

  Events.Draw("edmtestOtherThings_OtherThing_testUserTag_TEST.obj.ref.get().a")

This runs Draw through a Ref and navigates thinned collections using
the BareRootProductGetter:

  Events.Draw("edmtestTrackOfThingss_trackOfThingsProducerG__TEST.obj.ref1.get().a")
  Events.Draw("edmtestTrackOfThingss_trackOfThingsProducerDMinus__TEST.obj.ref1.get().a")

I tried and failed to draw through a Ref using the TBrowser,
although maybe there is some way to do it.

I also tried and failed to draw through a RefVector or PtrVector.
Again, there may be some way to do this and I just do not understand
the syntax. The BareRootProductGetter should support this without any
problems (the automated test below verifies this). I am not sure
whether ROOT can handle the complexities of navigating a PtrVector
using ROOT's "Draw" interface.

 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <vector>
#include "catch2/catch_all.hpp"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TSystem.h"
#include "TChain.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "DataFormats/Provenance/interface/BranchType.h"

class testRefInROOT {
public:
  testRefInROOT() {}
  ~testRefInROOT() {}
  void setUp() {
    if (!sWasRun_) {
      gSystem->Load("libFWCoreFWLite.so");
      FWLiteEnabler::enable();
      sWasRun_ = true;
    }
  }

  void testOneGoodFile();
  void testTwoGoodFiles();
  void failOneBadFile();
  void testGoodChain();
  // void failChainWithMissingFile();
  void failDidNotCallGetEntryForEvents();
  void testMissingRef();

private:
  static bool sWasRun_;
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
    //I'm assuming the following is true
    REQUIRE(itOther->ref.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->ref.get()->a != itThing->a) {
      std::cout << " *PROBLEM: ref " << itOther->ref.get()->a << "!= thing " << itThing->a << std::endl;
    }
    REQUIRE(itOther->ref.get()->a == itThing->a);

    REQUIRE(itOther->refToBase.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->refToBase.get()->a != itThing->a) {
      std::cout << " *PROBLEM: refToBase " << itOther->refToBase.get()->a << "!= thing " << itThing->a << std::endl;
    }
    REQUIRE(itOther->refToBase.get()->a == itThing->a);

    REQUIRE(itOther->ptr.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->ptr.get()->a != itThing->a) {
      std::cout << " *PROBLEM: ptr " << itOther->ptr.get()->a << "!= thing " << itThing->a << std::endl;
    }
    REQUIRE(itOther->ptr.get()->a == itThing->a);

    if (itView->a != itThing->a) {
      std::cout << " *PROBLEM: RefToBaseProd " << itView->a << "!= thing " << itThing->a << std::endl;
    }
    REQUIRE(itView->a == itThing->a);
  }
}

static void testTree(TTree* events) {
  REQUIRE(events != nullptr);

  /*
   edmtest::OtherThingCollection* pOthers = nullptr;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  REQUIRE(otherBranch != nullptr);

  //edmtest::ThingCollection things;
  edm::Wrapper<edmtest::ThingCollection>* pThings = nullptr;
  //NOTE: the period at the end is needed
  TBranch* thingBranch = events->GetBranch("edmtestThings_Thing__TEST.");
  REQUIRE(thingBranch != nullptr);

  int nev = events->GetEntries();
  for (int ev = 0; ev < nev; ++ev) {
    events->GetEntry(ev, 0);
    otherBranch->SetAddress(&pOthers);
    thingBranch->SetAddress(&pThings);
    thingBranch->GetEntry(ev);
    otherBranch->GetEntry(ev);
    checkMatch(pOthers->product(), pThings->product());
  }
}

void testRefInROOT::testOneGoodFile() {
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  testTree(events);
}

void testRefInROOT::failOneBadFile() {
  TFile file("thisFileDoesNotExist.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));

  REQUIRE(events == nullptr);
}

void testRefInROOT::testTwoGoodFiles() {
  std::cout << "gFile " << gFile << std::endl;
  TFile file("good.root");
  std::cout << " file :" << &file << " gFile: " << gFile << std::endl;

  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));

  testTree(events);
  std::cout << "working on second file" << std::endl;
  TFile file2("good2.root");
  std::cout << " file2 :" << &file2 << " gFile: " << gFile << std::endl;
  events = dynamic_cast<TTree*>(file2.Get(edm::poolNames::eventTreeName().c_str()));

  testTree(events);
}

void testRefInROOT::testGoodChain() {
  TChain eventChain(edm::poolNames::eventTreeName().c_str());
  eventChain.Add("good.root");
  eventChain.Add("good_delta5.root");
  eventChain.LoadTree(0);

  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* othersBranch = nullptr;
  eventChain.SetBranchAddress("edmtestOtherThings_OtherThing_testUserTag_TEST.", &pOthers, &othersBranch);

  edm::Wrapper<edmtest::ThingCollection>* pThings = nullptr;
  TBranch* thingsBranch = nullptr;
  eventChain.SetBranchAddress("edmtestThings_Thing__TEST.", &pThings, &thingsBranch);

  int nev = eventChain.GetEntries();
  for (int ev = 0; ev < nev; ++ev) {
    std::cout << "event #" << ev << std::endl;
    othersBranch->SetAddress(&pOthers);
    thingsBranch->SetAddress(&pThings);
    othersBranch->GetEntry(ev);
    thingsBranch->GetEntry(ev);
    eventChain.GetEntry(ev, 0);
    REQUIRE(pOthers != nullptr);
    REQUIRE(pThings != nullptr);
    checkMatch(pOthers->product(), pThings->product());
  }
}

void testRefInROOT::testMissingRef() {
  TFile file("other_only.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  REQUIRE(events != nullptr);
  if (events == nullptr)
    return;  // To silence Coverity

  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  REQUIRE(otherBranch != nullptr);

  int nev = events->GetEntries();
  for (int ev = 0; ev < nev; ++ev) {
    events->GetEntry(ev, 0);
    otherBranch->SetAddress(&pOthers);
    otherBranch->GetEntry(ev);

    for (auto const& prod : *pOthers->product()) {
      REQUIRE(not prod.ref.isAvailable());
      REQUIRE_THROWS_AS(prod.ref.get(), cms::Exception);
    }
  }
}

/*
void testRefInROOT::failChainWithMissingFile()
{
  TChain eventChain(edm::poolNames::eventTreeName().c_str());
  eventChain.Add("good.root");
  eventChain.Add("thisFileDoesNotExist.root");
  
  edm::Wrapper<edmtest::OtherThingCollection> *pOthers = nullptr;
  eventChain.SetBranchAddress("edmtestOtherThings_OtherThing_testUserTag_TEST.",&pOthers);
  
  edm::Wrapper<edmtest::ThingCollection>* pThings = nullptr;
  eventChain.SetBranchAddress("edmtestThings_Thing__TEST.",&pThings);
  
  int nev = eventChain.GetEntries();
  for( int ev=0; ev<nev; ++ev) {
    std::cout <<"event #" <<ev<<std::endl;    
    eventChain.GetEntry(ev);
    REQUIRE(pOthers != nullptr);
    REQUIRE(pThings != nullptr);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}
*/
void testRefInROOT::failDidNotCallGetEntryForEvents() {
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  REQUIRE(events != nullptr);
  if (events == nullptr)
    return;  // To silence Coverity

  /*
   edmtest::OtherThingCollection* pOthers = nullptr;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  REQUIRE(otherBranch != nullptr);
  otherBranch->SetAddress(&pOthers);

  otherBranch->GetEntry(0);

  REQUIRE(pOthers->product() != nullptr);

  pOthers->product()->at(0).ref.get();
}

TEST_CASE("ref_t", "[FWLite]") {
  testRefInROOT test;
  test.setUp();

  SECTION("testOneGoodFile") { test.testOneGoodFile(); }
  SECTION("failOneBadFile") { test.failOneBadFile(); }
  SECTION("testGoodChain") { test.testGoodChain(); }
  SECTION("testTwoGoodFiles") { test.testTwoGoodFiles(); }
  SECTION("testMissingRef") { test.testMissingRef(); }
  SECTION("failDidNotCallGetEntryForEvents") {
    REQUIRE_THROWS_AS(test.failDidNotCallGetEntryForEvents(), std::exception);
  }
}
