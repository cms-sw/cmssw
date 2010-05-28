/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT.

$Id: ref_t.cppunit.cpp,v 1.19 2009/03/02 20:34:34 wmtan Exp $
 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TSystem.h"
#include "TChain.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/TestHelper.h"

static char* gArgV = 0;

extern "C" char** environ;

#define CHARSTAR(x) const_cast<char *>(x)

class testRefInROOT: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testRefInROOT);
  
  CPPUNIT_TEST(testOneGoodFile);
  CPPUNIT_TEST_EXCEPTION(failOneBadFile,std::exception);
  CPPUNIT_TEST(testGoodChain);
  CPPUNIT_TEST(testTwoGoodFiles);
  CPPUNIT_TEST(testMissingRef);

  // CPPUNIT_TEST_EXCEPTION(failChainWithMissingFile,std::exception);
  //failTwoDifferentFiles
  CPPUNIT_TEST_EXCEPTION(failDidNotCallGetEntryForEvents,std::exception);
  
  CPPUNIT_TEST_SUITE_END();
public:
  testRefInROOT() { }
  void setUp()
  {
    if(!sWasRun_) {
      gSystem->Load("libFWCoreFWLite.so");
      AutoLibraryLoader::enable();
      
      char* argv[] = {CHARSTAR("testFWCoreFWLite"),
		      CHARSTAR("/bin/bash"),
		      CHARSTAR("FWCore/FWLite/test"),
		      CHARSTAR("RefTest.sh")};
      argv[0] = gArgV;
      if(0!=ptomaine(sizeof(argv)/sizeof(const char*), argv, environ) ) {
        std::cerr <<"could not run script needed to make test files\n";
        ::exit(-1);
      }
      sWasRun_ = true;
    }
  }
  void tearDown(){}
  
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

bool testRefInROOT::sWasRun_=false;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRefInROOT);

static void checkMatch(const edmtest::OtherThingCollection* pOthers,
                       const edmtest::ThingCollection* pThings)
{
  CPPUNIT_ASSERT(pOthers != 0);
  CPPUNIT_ASSERT(pThings != 0);
  CPPUNIT_ASSERT(pOthers->size() == pThings->size());
  
  //This test requires at least one entry
  CPPUNIT_ASSERT(pOthers->size() > 0 );
  const edm::View<edmtest::Thing>& view = *(pOthers->front().refToBaseProd);
  CPPUNIT_ASSERT(view.size() == pOthers->size());
  
  edmtest::ThingCollection::const_iterator itThing = pThings->begin(), itThingEnd = pThings->end();
  edmtest::OtherThingCollection::const_iterator itOther = pOthers->begin();
  edm::View<edmtest::Thing>::const_iterator itView = view.begin();
  
  for( ; itThing != itThingEnd; ++itThing, ++itOther,++itView) {
    //I'm assuming the following is true
    CPPUNIT_ASSERT(itOther->ref.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if(itOther->ref.get()->a != itThing->a) {
      std::cout <<" *PROBLEM: ref "<<itOther->ref.get()->a<<"!= thing "<<itThing->a<<std::endl;
    }
    CPPUNIT_ASSERT( itOther->ref.get()->a == itThing->a);

    CPPUNIT_ASSERT(itOther->refToBase.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if(itOther->refToBase.get()->a != itThing->a) {
      std::cout <<" *PROBLEM: refToBase "<<itOther->refToBase.get()->a<<"!= thing "<<itThing->a<<std::endl;
    }
    CPPUNIT_ASSERT( itOther->refToBase.get()->a == itThing->a);

    CPPUNIT_ASSERT(itOther->ptr.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    //std::cout <<" ref "<<itOther->ref.get()->a<<" thing "<<itThing->a<<std::endl;
    if(itOther->ptr.get()->a != itThing->a) {
      std::cout <<" *PROBLEM: ptr "<<itOther->ptr.get()->a<<"!= thing "<<itThing->a<<std::endl;
    }
    CPPUNIT_ASSERT( itOther->ptr.get()->a == itThing->a);
    
    
    if(itView->a != itThing->a) {
      std::cout <<" *PROBLEM: RefToBaseProd "<<itView->a<<"!= thing "<<itThing->a<<std::endl;
    }
    CPPUNIT_ASSERT( itView->a == itThing->a);
  }
}

static void testTree(TTree* events) {
  CPPUNIT_ASSERT(events !=0);
  
  /*
   edmtest::OtherThingCollection* pOthers = 0;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.obj");
   this does NOT work, must get the wrapper
   */
  edm::Wrapper<edmtest::OtherThingCollection> *pOthers = 0;
  TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");
  
  CPPUNIT_ASSERT( otherBranch != 0);
  
  //edmtest::ThingCollection things;
  edm::Wrapper<edmtest::ThingCollection>* pThings = 0;
  //NOTE: the period at the end is needed
  TBranch* thingBranch = events->GetBranch("edmtestThings_Thing__TEST.");
  CPPUNIT_ASSERT( thingBranch != 0);
  
  int nev = events->GetEntries();
  for( int ev=0; ev<nev; ++ev) {

    events->GetEntry(ev,0);
    otherBranch->SetAddress(&pOthers);
    thingBranch->SetAddress(&pThings);
    thingBranch->GetEntry(ev);
    otherBranch->GetEntry(ev);
    checkMatch(pOthers->product(),pThings->product());
  }
}

void testRefInROOT::testOneGoodFile()
{
   TFile file("good.root");
   TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
   
   testTree(events);
}

void testRefInROOT::failOneBadFile()
{
  TFile file("thisFileDoesNotExist.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  
  testTree(events);
}

void testRefInROOT::testTwoGoodFiles()
{
  std::cout <<"gFile "<<gFile<<std::endl;
  TFile file("good.root");
  std::cout <<" file :" << &file<<" gFile: "<<gFile<<std::endl;
  
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
  
  testTree(events);
  std::cout <<"working on second file"<<std::endl;
  TFile file2("good2.root");
  std::cout <<" file2 :"<< &file2<<" gFile: "<<gFile<<std::endl;
  events = dynamic_cast<TTree*>(file2.Get(edm::poolNames::eventTreeName().c_str()));
  
  testTree(events);
}


void testRefInROOT::testGoodChain()
{
  TChain eventChain(edm::poolNames::eventTreeName().c_str());
  eventChain.Add("good.root");
  eventChain.Add("good_delta5.root");

  edm::Wrapper<edmtest::OtherThingCollection> *pOthers =0;
  TBranch* othersBranch = 0;
  eventChain.SetBranchAddress("edmtestOtherThings_OtherThing_testUserTag_TEST.",&pOthers,&othersBranch);
  
  edm::Wrapper<edmtest::ThingCollection>* pThings = 0;
  TBranch* thingsBranch = 0;
  eventChain.SetBranchAddress("edmtestThings_Thing__TEST.",&pThings,&thingsBranch);
  
  int nev = eventChain.GetEntries();
  for( int ev=0; ev<nev; ++ev) {
    std::cout <<"event #" <<ev<<std::endl;
    othersBranch->SetAddress(&pOthers);
    thingsBranch->SetAddress(&pThings);
    othersBranch->GetEntry(ev);
    thingsBranch->GetEntry(ev);
    eventChain.GetEntry(ev,0);
    CPPUNIT_ASSERT(pOthers != 0);
    CPPUNIT_ASSERT(pThings != 0);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}

void testRefInROOT::testMissingRef()
{
   TFile file("other_only.root");
   TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
   
   edm::Wrapper<edmtest::OtherThingCollection> *pOthers = 0;
   TBranch* otherBranch = events->GetBranch("edmtestOtherThings_OtherThing_testUserTag_TEST.");
   
   CPPUNIT_ASSERT( otherBranch != 0);
   
   int nev = events->GetEntries();
   for( int ev=0; ev<nev; ++ev) {
      
      events->GetEntry(ev,0);
      otherBranch->SetAddress(&pOthers);
      otherBranch->GetEntry(ev);
      
      for(edmtest::OtherThingCollection::const_iterator itOther = pOthers->product()->begin(), 
          itEnd=pOthers->product()->end();
          itOther != itEnd; ++itOther) {
         CPPUNIT_ASSERT(not itOther->ref.isAvailable());
         CPPUNIT_ASSERT_THROW(itOther->ref.get(), cms::Exception);
      }
      
   }   
}

/*
void testRefInROOT::failChainWithMissingFile()
{
  TChain eventChain(edm::poolNames::eventTreeName().c_str());
  eventChain.Add("good.root");
  eventChain.Add("thisFileDoesNotExist.root");
  
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
void testRefInROOT::failDidNotCallGetEntryForEvents()
{
  TFile file("good.root");
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName().c_str()));
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

//Stolen from Utilities/Testing/interface/CppUnit_testdriver.icpp
// need to refactor
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextTestProgressListener.h>
#include <stdexcept>  

int 
main( int argc, char* argv[] )
{
  gArgV = argv[0];
  std::string testPath = (argc > 1) ? std::string(argv[1]) : "";
  
  // Create the event manager and test controller
  CppUnit::TestResult controller;
  
  // Add a listener that colllects test result
  CppUnit::TestResultCollector result;
  controller.addListener( &result );        
  
  // Add a listener that print dots as test run.
  CppUnit::TextTestProgressListener progress;
  controller.addListener( &progress );      
  
  // Add the top suite to the test runner
  CppUnit::TestRunner runner;
  runner.addTest( CppUnit::TestFactoryRegistry::getRegistry().makeTest() );   
  try
  {
    std::cout << "Running "  <<  testPath;
    runner.run( controller, testPath );
    
    std::cerr << std::endl;
    
    // Print test in a compiler compatible format.
    CppUnit::CompilerOutputter outputter( &result, std::cerr );
    outputter.write();                      
  }
  catch ( std::invalid_argument &e )  // Test path not resolved
  {
    std::cerr  <<  std::endl  
    <<  "ERROR: "  <<  e.what()
    << std::endl;
    return 0;
  }
  
  return result.wasSuccessful() ? 0 : 1;
}

