/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT.

$Id: test.cppunit.cpp,v 1.2 2007/05/16 16:48:33 chrjones Exp $
 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TChain.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Utilities/interface/TestHelper.h"

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
static char* gArgV = 0;

class testRefInROOT: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testRefInROOT);
  
  CPPUNIT_TEST(testOneGoodFile);
  CPPUNIT_TEST_EXCEPTION(failOneBadFile,std::exception);
  CPPUNIT_TEST(testRefFirst);
  CPPUNIT_TEST(testAllLabels);
  CPPUNIT_TEST(testGoodChain);
  CPPUNIT_TEST(testTwoGoodFiles);
  // CPPUNIT_TEST_EXCEPTION(failChainWithMissingFile,std::exception);
  //failTwoDifferentFiles
  //CPPUNIT_TEST_EXCEPTION(failDidNotCallGetEntryForEvents,std::exception);
  
  CPPUNIT_TEST_SUITE_END();
public:
  testRefInROOT() { }
  void setUp()
  {
    if(!sWasRun_) {
      gSystem->Load("libFWCoreFWLite.so");
      AutoLibraryLoader::enable();
      
      char* argv[] = {"TestRunnerDataFormatsFWLite","/bin/bash","DataFormats/FWLite/test","RefTest.sh"};
      argv[0] = gArgV;
      if(0!=ptomaine(sizeof(argv)/sizeof(const char*), argv) ) {
        std::cerr <<"could not run script needed to make test files\n";
        ::exit(-1);
      }
      sWasRun_ = true;
    }
  }
  void tearDown(){}
  
  void testRefFirst();
  void testAllLabels();
  void testOneGoodFile();
  void testTwoGoodFiles();
  void failOneBadFile();
  void testGoodChain();
  // void failChainWithMissingFile();
  //void failDidNotCallGetEntryForEvents();

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
    //std::cout <<"getting data"<<std::endl;
    //I'm assuming the following is true
    CPPUNIT_ASSERT(itOther->ref.key() == static_cast<unsigned int>(itThing - pThings->begin()));
    CPPUNIT_ASSERT( itOther->ref.get()->a == itThing->a);
    if(itView->a != itThing->a) {
      std::cout <<" *PROBLEM: RefToBaseProd "<<itView->a<<"!= thing "<<itThing->a<<std::endl;
    }
    CPPUNIT_ASSERT( itView->a == itThing->a);
  }
}

static void testEvent(fwlite::Event& events) {
  
  for(events.toBegin(); not events.atEnd(); ++events) {
    fwlite::Handle<edmtest::ThingCollection> pThings ;
    pThings.getByLabel(events,"Thing");

    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events,"OtherThing","testUserTag");

    checkMatch(pOthers.ptr(),pThings.ptr());
  }
}

void testRefInROOT::testOneGoodFile()
{
   TFile file("good.root");
   fwlite::Event events(&file);
   
   testEvent(events);
}

void testRefInROOT::testAllLabels()
{
  TFile file("good.root");
  fwlite::Event events(&file);

  for(events.toBegin(); not events.atEnd(); ++events) {
    
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events,"OtherThing","testUserTag","TEST");
  }
}

void testRefInROOT::testRefFirst()
{
  TFile file("good.root");
  fwlite::Event events(&file);
  
  for(events.toBegin(); not events.atEnd(); ++events) {
    
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events,"OtherThing","testUserTag");

    //std::cout <<"got OtherThing"<<std::endl;
    for(edmtest::OtherThingCollection::const_iterator itOther=pOthers->begin(), itEnd=pOthers->end() ;
        itOther != itEnd; ++itOther) {
      //std::cout <<"getting ref"<<std::endl;
      itOther->ref.get()->a;
    }
    //std::cout <<"get all Refs"<<std::endl;
    
    fwlite::Handle<edmtest::ThingCollection> pThings ;
    pThings.getByLabel(events,"Thing");

    //std::cout <<"checkMatch"<<std::endl;
    checkMatch(pOthers.ptr(),pThings.ptr());
  }
}

void testRefInROOT::failOneBadFile()
{
  TFile file("thisFileDoesNotExist.root");
  fwlite::Event events(&file);
  
  testEvent(events);
}

void testRefInROOT::testTwoGoodFiles()
{
  /*
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
   */
}


void testRefInROOT::testGoodChain()
{
  /*
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
  */
}
/*
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
    std::cout <<"event #" <<ev<<std::endl;    
    eventChain.GetEntry(ev);
    CPPUNIT_ASSERT(pOthers != 0);
    CPPUNIT_ASSERT(pThings != 0);
    checkMatch(pOthers->product(),pThings->product());
  }
  
}
*/

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

