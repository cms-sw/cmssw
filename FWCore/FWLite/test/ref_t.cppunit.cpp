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
  CPPUNIT_TEST(testThinning);

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
  void testThinning();

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

void testRefInROOT::testThinning() {
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  CPPUNIT_ASSERT(events != nullptr);
  if (events == nullptr)
    return;  // To silence Coverity
  edm::Wrapper<std::vector<edmtest::TrackOfThings> >* pTracks = nullptr;
  TBranch* tracksBranchD = events->GetBranch("edmtestTrackOfThingss_trackOfThingsProducerDPlus__TEST.");
  TBranch* tracksBranchG = events->GetBranch("edmtestTrackOfThingss_trackOfThingsProducerG__TEST.");
  TBranch* tracksBranchM = events->GetBranch("edmtestTrackOfThingss_trackOfThingsProducerM__TEST.");
  CPPUNIT_ASSERT(tracksBranchD != nullptr && tracksBranchG != nullptr && tracksBranchM != nullptr);

  std::vector<edmtest::TrackOfThings> const* vTracks = nullptr;

  int nev = events->GetEntries();
  for (int ev = 0; ev < nev; ++ev) {
    // The values in the tests below have no particular meaning.
    // It is just checking that we read the values known to be
    // be put in by the relevant producer.

    int offset = 200 + ev * 100;

    events->GetEntry(ev, 0);

    // In the D branch this tests accessing a value in
    // thinned collection made from a thinned collection
    // made from a master collection.
    tracksBranchD->SetAddress(&pTracks);
    tracksBranchD->GetEntry(ev);
    vTracks = pTracks->product();
    CPPUNIT_ASSERT(vTracks != nullptr);
    edmtest::TrackOfThings const& trackD = vTracks->at(0);
    CPPUNIT_ASSERT(trackD.ref1.isAvailable());
    CPPUNIT_ASSERT(trackD.ref1->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackD.ptr1->a == 12 + offset);
    CPPUNIT_ASSERT(trackD.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackD.refToBase1->a == 10 + offset);

    CPPUNIT_ASSERT(trackD.refVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.refVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.refVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackD.refVector1.isAvailable());
    CPPUNIT_ASSERT(trackD.refVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.refVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.refVector1[8].operator->(), cms::Exception);

    CPPUNIT_ASSERT(trackD.ptrVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptrVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.ptrVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackD.ptrVector1[9]->a == 21 + offset);
    CPPUNIT_ASSERT(!trackD.ptrVector1.isAvailable());
    CPPUNIT_ASSERT(trackD.ptrVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptrVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.ptrVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackD.ptrVector1[9]->a == 21 + offset);

    CPPUNIT_ASSERT(trackD.refToBaseVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.refToBaseVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.refToBaseVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackD.refToBaseVector1.isAvailable());
    CPPUNIT_ASSERT(trackD.refToBaseVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.refToBaseVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.refToBaseVector1[8].operator->(), cms::Exception);

    // In the G branch this tests accessing a value in
    // thinned collection made from a master collection.
    // Otherwise the tests are very similar the preceding
    // tests.
    tracksBranchG->SetAddress(&pTracks);
    tracksBranchG->GetEntry(ev);
    vTracks = pTracks->product();
    CPPUNIT_ASSERT(vTracks != nullptr);
    edmtest::TrackOfThings const& trackG = vTracks->at(0);
    CPPUNIT_ASSERT(trackG.ref1.isAvailable());
    CPPUNIT_ASSERT(trackG.ref1->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackG.ptr1->a == 22 + offset);
    CPPUNIT_ASSERT(trackG.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackG.refToBase1->a == 20 + offset);

    CPPUNIT_ASSERT(trackG.refVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.refVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.refVector1[8]->a == 28 + offset);
    CPPUNIT_ASSERT(trackG.refVector1.isAvailable());
    CPPUNIT_ASSERT(trackG.refVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.refVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.refVector1[8]->a == 28 + offset);

    CPPUNIT_ASSERT(trackG.ptrVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[8]->a == 28 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1.isAvailable());
    CPPUNIT_ASSERT(trackG.ptrVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[8]->a == 28 + offset);

    CPPUNIT_ASSERT(trackG.refToBaseVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.refToBaseVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.refToBaseVector1[8]->a == 28 + offset);
    CPPUNIT_ASSERT(trackG.refToBaseVector1.isAvailable());
    CPPUNIT_ASSERT(trackG.refToBaseVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.refToBaseVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.refToBaseVector1[8]->a == 28 + offset);

    // The tests for the M branch are very similar to the preceding
    // tests except some of the elements are through two levels
    // of thinning or some just one level of thinning.
    tracksBranchM->SetAddress(&pTracks);
    tracksBranchM->GetEntry(ev);
    vTracks = pTracks->product();
    CPPUNIT_ASSERT(vTracks != nullptr);

    edmtest::TrackOfThings const& trackM0 = vTracks->at(0);
    CPPUNIT_ASSERT(!trackM0.ref1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.ref1.operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM0.ptr1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.ptr1.operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM0.refToBase1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.refToBase1.operator->(), cms::Exception);

    edmtest::TrackOfThings const& trackM1 = vTracks->at(1);
    CPPUNIT_ASSERT(trackM1.ref1.isAvailable());
    CPPUNIT_ASSERT(trackM1.ref1->a == 44 + offset);
    CPPUNIT_ASSERT(trackM1.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackM1.ptr1->a == 46 + offset);
    CPPUNIT_ASSERT(trackM1.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackM1.refToBase1->a == 44 + offset);

    edmtest::TrackOfThings const& trackM = vTracks->at(0);
    CPPUNIT_ASSERT_THROW(trackM.refVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.refVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.refVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM.refVector1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM.refVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.refVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.refVector1[8].operator->(), cms::Exception);

    CPPUNIT_ASSERT_THROW(trackM.ptrVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.ptrVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.ptrVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM.ptrVector1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM.ptrVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.ptrVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.ptrVector1[8].operator->(), cms::Exception);

    CPPUNIT_ASSERT_THROW(trackM.refToBaseVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.refToBaseVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.refToBaseVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM.refToBaseVector1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM.refToBaseVector1[0].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackM.refToBaseVector1[4]->a == 44 + offset);
    CPPUNIT_ASSERT_THROW(trackM.refToBaseVector1[8].operator->(), cms::Exception);
  }
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
