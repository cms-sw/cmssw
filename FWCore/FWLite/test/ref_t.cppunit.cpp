/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT.

$Id$
 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/FWLite/src/AutoLibraryLoader.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TChain.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

class testRefInROOT: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testRefInROOT);
  
  CPPUNIT_TEST(testOneGoodFile);
  CPPUNIT_TEST_EXCEPTION(failOneBadFile,std::exception);
  //CPPUNIT_TEST(testGoodChain);
  CPPUNIT_TEST(testTwoGoodFiles);
  CPPUNIT_TEST_EXCEPTION(failChainWithMissingFile,std::exception);
  //failTwoDifferentFiles
  CPPUNIT_TEST_EXCEPTION(failDidNotCallGetEntryForEvents,std::exception);
  
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp()
  {
      gSystem->Load("libFWCoreFWLite.so");
      AutoLibraryLoader::enable();
  }
  void tearDown(){}
  
  void testOneGoodFile();
  void testTwoGoodFiles();
  void failOneBadFile();
  //void testGoodChain();
  void failChainWithMissingFile();
  void failDidNotCallGetEntryForEvents();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRefInROOT);

using std::cout;
using std::endl;

static void checkMatch(const edmtest::OtherThingCollection* pOthers,
                       const edmtest::ThingCollection* pThings)
{
  CPPUNIT_ASSERT(pOthers != 0);
  CPPUNIT_ASSERT(pThings !=0);
  CPPUNIT_ASSERT(pOthers->size() == pThings->size());
  
  edmtest::ThingCollection::const_iterator itThing=pThings->begin();
  edmtest::OtherThingCollection::const_iterator itOther=pOthers->begin();
  
  for( ; itThing != pThings->end(); ++itThing,++itOther) {
    //I'm assuming the following is true
    CPPUNIT_ASSERT(itOther->ref.index() == static_cast<unsigned long>(itThing - pThings->begin()));
    CPPUNIT_ASSERT( itOther->ref.get()->a == itThing->a);
  }
}

static void testTree(TTree* events) {
  CPPUNIT_ASSERT(events !=0);
  
  /*
   edmtest::OtherThingCollection* pOthers = 0;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection> *pOthers =0;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");
  
  CPPUNIT_ASSERT( otherBranch != 0);
  otherBranch->SetAddress(&pOthers);
  
  //edmtest::ThingCollection things;
  edm::Wrapper<edmtest::ThingCollection>* pThings = 0;
  //NOTE: the period at the end is needed
  TBranch* thingBranch = events->GetBranch("edmtestThings_Thing__TEST.");
  CPPUNIT_ASSERT( thingBranch != 0);
  thingBranch->SetAddress(&pThings);
  
  int nev = events->GetEntries();
  for( int ev=0; ev<nev; ++ev) {

    events->GetEntry(ev);
    thingBranch->GetEntry(ev);
    otherBranch->GetEntry(ev);
    checkMatch(pOthers->product(),pThings->product());
  }
}

void testRefInROOT::testOneGoodFile()
{
   TFile file("good.root");
   TTree* events = dynamic_cast<TTree*>(file.Get("Events"));
   
   testTree(events);
}

void testRefInROOT::failOneBadFile()
{
  TFile file("thisFileDoesNotExist.root");
  TTree* events = dynamic_cast<TTree*>(file.Get("Events"));
  
  testTree(events);
}

void testRefInROOT::testTwoGoodFiles()
{
  std::cout <<"gFile "<<gFile<<std::endl;
  TFile file("good.root");
  std::cout <<" file :" << &file<<" gFile: "<<gFile<<std::endl;
  
  TTree* events = dynamic_cast<TTree*>(file.Get("Events"));
  
  testTree(events);
  std::cout <<"working on second file"<<std::endl;
  TFile file2("good2.root");
  std::cout <<" file2 :"<< &file2<<" gFile: "<<gFile<<std::endl;
  events = dynamic_cast<TTree*>(file2.Get("Events"));
  
  testTree(events);
}

/*
void testRefInROOT::testGoodChain()
{
  TChain eventChain("Events");
  eventChain.Add("good.root");
  eventChain.Add("good2.root");

  edm::Wrapper<edmtest::OtherThingCollection> *pOthers =0;
  eventChain.SetBranchAddress("edmtestOtherThings_OtherThing_testUserTag_TEST.",&pOthers);
  
  edm::Wrapper<edmtest::ThingCollection>* pThings = 0;
  eventChain.SetBranchAddress("edmtestThings_Thing__TEST.",&pThings);
  
  int nev = eventChain.GetEntries();
  for( int ev=0; ev<nev; ++ev) {
    std::cout <<"event #" <<ev<<std::endl;
    eventChain.GetEntry(ev);
    CPPUNIT_ASSERT(pOthers != 0);
    CPPUNIT_ASSERT(pThings != 0);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}
*/
void testRefInROOT::failChainWithMissingFile()
{
  TChain eventChain("Events");
  eventChain.Add("good.root");
  eventChain.Add("thisFileDoesNotExist.root");
  
  edm::Wrapper<edmtest::OtherThingCollection> *pOthers =0;
  eventChain.SetBranchAddress("edmtestOtherThings_OtherThing_testUserTag_TEST.",&pOthers);
  
  edm::Wrapper<edmtest::ThingCollection>* pThings = 0;
  eventChain.SetBranchAddress("edmtestThings_Thing__TEST.",&pThings);
  
  int nev = eventChain.GetEntries();
  for( int ev=0; ev<nev; ++ev) {
    
    eventChain.GetEntry(ev);
    CPPUNIT_ASSERT(pOthers != 0);
    CPPUNIT_ASSERT(pThings != 0);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}

void testRefInROOT::failDidNotCallGetEntryForEvents()
{
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get("Events"));
  CPPUNIT_ASSERT(events !=0);
  
  /*
   edmtest::OtherThingCollection* pOthers = 0;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection> *pOthers =0;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");
  
  CPPUNIT_ASSERT( otherBranch != 0);
  otherBranch->SetAddress(&pOthers);
  
  otherBranch->GetEntry(0);
  
  CPPUNIT_ASSERT(pOthers->product() != 0);

  pOthers->product()->at(0).ref.get();
}
