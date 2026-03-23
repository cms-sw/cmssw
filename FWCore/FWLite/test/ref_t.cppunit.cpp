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
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TSystem.h"
#include "TChain.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "DataFormats/Provenance/interface/BranchType.h"

class testRefInROOT : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRefInROOT);

  CPPUNIT_TEST(testOneGoodFile);
  CPPUNIT_TEST_EXCEPTION(failOneBadFile, std::exception);
  CPPUNIT_TEST(testGoodChain);
  CPPUNIT_TEST(testTwoGoodFiles);
  CPPUNIT_TEST(testMissingRef);

  // CPPUNIT_TEST_EXCEPTION(failChainWithMissingFile,std::exception);
  //failTwoDifferentFiles
  CPPUNIT_TEST_EXCEPTION(failDidNotCallGetEntryForEvents, std::exception);

  CPPUNIT_TEST_SUITE_END();

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
  void tearDown() {}

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

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRefInROOT);

static void checkMatch(const edmtest::OtherThingCollection* pOthers, const edmtest::ThingCollection* pThings) {
  CPPUNIT_ASSERT(pOthers != nullptr);
  CPPUNIT_ASSERT(pThings != nullptr);
  CPPUNIT_ASSERT(pOthers->size() == pThings->size());

  //This test requires at least one entry
  CPPUNIT_ASSERT(pOthers->size() > 0);
  const edm::View<edmtest::Thing>& view = *(pOthers->front().refToBaseProd);
  CPPUNIT_ASSERT(view.size() == pOthers->size());

  edmtest::ThingCollection::const_iterator itThing = pThings->begin(), itThingEnd = pThings->end();
  edmtest::OtherThingCollection::const_iterator itOther = pOthers->begin();
  edm::View<edmtest::Thing>::const_iterator itView = view.begin();

  for (; itThing != itThingEnd; ++itThing, ++itOther, ++itView) {
    //I'm assuming the following is true
    CPPUNIT_ASSERT(itOther->ref.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->ref.get()->a != itThing->a) {
      std::cout << " *PROBLEM: ref " << itOther->ref.get()->a << "!= thing " << itThing->a << std::endl;
    }
    CPPUNIT_ASSERT(itOther->ref.get()->a == itThing->a);

    CPPUNIT_ASSERT(itOther->refToBase.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->refToBase.get()->a != itThing->a) {
      std::cout << " *PROBLEM: refToBase " << itOther->refToBase.get()->a << "!= thing " << itThing->a << std::endl;
    }
    CPPUNIT_ASSERT(itOther->refToBase.get()->a == itThing->a);

    CPPUNIT_ASSERT(itOther->ptr.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if (itOther->ptr.get()->a != itThing->a) {
      std::cout << " *PROBLEM: ptr " << itOther->ptr.get()->a << "!= thing " << itThing->a << std::endl;
    }
    CPPUNIT_ASSERT(itOther->ptr.get()->a == itThing->a);

    if (itView->a != itThing->a) {
      std::cout << " *PROBLEM: RefToBaseProd " << itView->a << "!= thing " << itThing->a << std::endl;
    }
    CPPUNIT_ASSERT(itView->a == itThing->a);
  }
}

static void testTree(TTree* events) {
  CPPUNIT_ASSERT(events != 0);

  /*
   edmtest::OtherThingCollection* pOthers = nullptr;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  CPPUNIT_ASSERT(otherBranch != nullptr);

  //edmtest::ThingCollection things;
  edm::Wrapper<edmtest::ThingCollection>* pThings = nullptr;
  //NOTE: the period at the end is needed
  TBranch* thingBranch = events->GetBranch("edmtestThings_Thing__TEST.");
  CPPUNIT_ASSERT(thingBranch != nullptr);

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

  testTree(events);
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
    CPPUNIT_ASSERT(pOthers != nullptr);
    CPPUNIT_ASSERT(pThings != nullptr);
    checkMatch(pOthers->product(), pThings->product());
  }
}

void testRefInROOT::testMissingRef() {
  TFile file("other_only.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  CPPUNIT_ASSERT(events != nullptr);
  if (events == nullptr)
    return;  // To silence Coverity

  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  CPPUNIT_ASSERT(otherBranch != nullptr);

  int nev = events->GetEntries();
  for (int ev = 0; ev < nev; ++ev) {
    events->GetEntry(ev, 0);
    otherBranch->SetAddress(&pOthers);
    otherBranch->GetEntry(ev);

    for (auto const& prod : *pOthers->product()) {
      CPPUNIT_ASSERT(not prod.ref.isAvailable());
      CPPUNIT_ASSERT_THROW(prod.ref.get(), cms::Exception);
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
    CPPUNIT_ASSERT(pOthers != nullptr);
    CPPUNIT_ASSERT(pThings != nullptr);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}
*/
void testRefInROOT::failDidNotCallGetEntryForEvents() {
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  CPPUNIT_ASSERT(events != nullptr);
  if (events == nullptr)
    return;  // To silence Coverity

  /*
   edmtest::OtherThingCollection* pOthers = nullptr;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection>* pOthers = nullptr;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");

  CPPUNIT_ASSERT(otherBranch != nullptr);
  otherBranch->SetAddress(&pOthers);

  otherBranch->GetEntry(0);

  CPPUNIT_ASSERT(pOthers->product() != nullptr);

  pOthers->product()->at(0).ref.get();
}

//Stolen from Utilities/Testing/interface/CppUnit_testdriver.icpp
// need to refactor
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextTestProgressListener.h>
#include <stdexcept>

int main() {
  // Create the event manager and test controller
  CppUnit::TestResult controller;

  // Add a listener that colllects test result
  CppUnit::TestResultCollector result;
  controller.addListener(&result);

  // Add a listener that print dots as test run.
  CppUnit::TextTestProgressListener progress;
  controller.addListener(&progress);

  // Add the top suite to the test runner
  CppUnit::TestRunner runner;
  runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
  try {
    std::cout << "Running ";
    runner.run(controller);

    std::cerr << std::endl;

    // Print test in a compiler compatible format.
    CppUnit::CompilerOutputter outputter(&result, std::cerr);
    outputter.write();
  } catch (std::invalid_argument& e)  // Test path not resolved
  {
    std::cerr << std::endl << "ERROR: " << e.what() << std::endl;
    return 0;
  }

  return result.wasSuccessful() ? 0 : 1;
}
