/*----------------------------------------------------------------------

Test program for edm::Ref use in ROOT.

 ----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <vector>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "TFile.h"
#include "TSystem.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "FWCore/Utilities/interface/TestHelper.h"

#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/MultiChainEvent.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

static char* gArgV = 0;

extern "C" char** environ;

#define CHARSTAR(x) const_cast<char *>(x)

class testRefInROOT: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testRefInROOT);
  
  CPPUNIT_TEST(testOneGoodFile);
  CPPUNIT_TEST_EXCEPTION(failOneBadFile,std::exception);
  CPPUNIT_TEST(testRefFirst);
  CPPUNIT_TEST(testAllLabels);
  CPPUNIT_TEST(testGoodChain);
  CPPUNIT_TEST(testTwoGoodFiles);
   CPPUNIT_TEST(testHandleErrors);
   CPPUNIT_TEST(testMissingRef);
   CPPUNIT_TEST(testMissingData);
   CPPUNIT_TEST(testEventBase);
   CPPUNIT_TEST(testSometimesMissingData);
   CPPUNIT_TEST(testTo);
   CPPUNIT_TEST(testThinning);

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
      
      char* argv[] = {CHARSTAR("TestRunnerDataFormatsFWLite"),
		      CHARSTAR("/bin/bash"),
		      CHARSTAR("DataFormats/FWLite/test"),
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
  // void failChainWithMissingFile();
  //void failDidNotCallGetEntryForEvents();
  void testThinning();

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

void testRefInROOT::testEventBase()
{
   TFile file("good.root");
   fwlite::Event events(&file);
   edm::InputTag tagFull("OtherThing","testUserTag","TEST");
   edm::InputTag tag("OtherThing","testUserTag");
   edm::InputTag tagNotHere("NotHereOtherThing");
   edm::InputTag tagThing("Thing");
   edm::EventBase* eventBase = &events;
   
   for(events.toBegin(); not events.atEnd(); ++events) {

      {
         edm::Handle<edmtest::OtherThingCollection> pOthers;
         eventBase->getByLabel(tagFull,pOthers);
         CPPUNIT_ASSERT(pOthers.isValid());

         // Test that the get function that takes a ProductID works
         // by getting a ProductID from a Ref stored in the OtherThingCollection
         // and testing that one can retrieve the ThingCollection with it.
         CPPUNIT_ASSERT(pOthers->size() > 0 );
         edmtest::OtherThingCollection::const_iterator itOther = pOthers->begin();
         edm::ProductID thingProductID = itOther->ref.id();
         edm::Handle<edmtest::ThingCollection> thingCollectionHandle;
         eventBase->get(thingProductID, thingCollectionHandle);
         edm::Handle<edmtest::ThingCollection> thingCollectionHandle2;
         eventBase->getByLabel(tagThing, thingCollectionHandle2);
         CPPUNIT_ASSERT(thingCollectionHandle.product() == thingCollectionHandle2.product() &&
                        thingCollectionHandle.product()->begin()->a == thingCollectionHandle2.product()->begin()->a);
      }
      {
         edm::Handle<edmtest::OtherThingCollection> pOthers;
         eventBase->getByLabel(tag,pOthers);
         CPPUNIT_ASSERT(pOthers.isValid());
         pOthers->size();
      }

      {
         edm::Handle<edmtest::OtherThingCollection> pOthers;
         eventBase->getByLabel(tagNotHere,pOthers);
         
         CPPUNIT_ASSERT(not pOthers.isValid());
         CPPUNIT_ASSERT(pOthers.failedToGet());
         CPPUNIT_ASSERT_THROW(pOthers.product(), cms::Exception);
      }
      
   }
   
}

void testRefInROOT::testTo()
{
   TFile file("good.root");
   fwlite::Event events(&file);
   edm::InputTag tag("Thing");
   edm::EventBase* eventBase = &events;
   
   CPPUNIT_ASSERT(events.to(1,1,2));
   {
      edm::Handle<edmtest::ThingCollection> pThings;
      eventBase->getByLabel(tag,pThings);
      CPPUNIT_ASSERT(pThings.isValid());
      CPPUNIT_ASSERT(0!=pThings->size());
      CPPUNIT_ASSERT(3 == (*pThings)[0].a);
   }
   std::cout <<events.id()<<std::endl;
   CPPUNIT_ASSERT(edm::EventID(1,1,2)==events.id());
   
   CPPUNIT_ASSERT(events.to(1,1,1));
   {
      edm::Handle<edmtest::ThingCollection> pThings;
      eventBase->getByLabel(tag,pThings);
      CPPUNIT_ASSERT(pThings.isValid());
      CPPUNIT_ASSERT(0!=pThings->size());
      CPPUNIT_ASSERT(2 == (*pThings)[0].a);
   }
   CPPUNIT_ASSERT(edm::EventID(1,1,1)==events.id());
  
   CPPUNIT_ASSERT( events.to(1));
   CPPUNIT_ASSERT(not events.to(events.size()));
   
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
      int arbitraryBigNumber = 1000000; 
      CPPUNIT_ASSERT(itOther->ref.get()->a < arbitraryBigNumber);
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

void testRefInROOT::testMissingRef()
{
   TFile file("other_only.root");
   fwlite::Event events(&file);
   
   for(events.toBegin(); not events.atEnd(); ++events) {
      
      fwlite::Handle<edmtest::OtherThingCollection> pOthers;
      pOthers.getByLabel(events,"OtherThing","testUserTag");
      for(edmtest::OtherThingCollection::const_iterator itOther=pOthers->begin(), itEnd=pOthers->end() ;
          itOther != itEnd; ++itOther) {
         //std::cout <<"getting ref"<<std::endl;
         CPPUNIT_ASSERT(not itOther->ref.isAvailable());
         CPPUNIT_ASSERT_THROW(itOther->ref.get(), cms::Exception);
      }
   }
}

void testRefInROOT::testMissingData()
{
   TFile file("good.root");
   fwlite::Event events(&file);
   
   for(events.toBegin(); not events.atEnd(); ++events) {
      
      fwlite::Handle<edmtest::OtherThingCollection> pOthers;
      pOthers.getByLabel(events,"NotHereOtherThing");
      
      CPPUNIT_ASSERT(not pOthers.isValid());
      CPPUNIT_ASSERT(pOthers.failedToGet());
      CPPUNIT_ASSERT_THROW(pOthers.product(), cms::Exception);
   }
}

void testRefInROOT::testSometimesMissingData()
{
  TFile file("partialEvent.root");
  fwlite::Event events(&file);
  
  unsigned int index=0;
  edm::InputTag tag("OtherThing","testUserTag");
  for(events.toBegin(); not events.atEnd(); ++events,++index) {
    
    fwlite::Handle<edmtest::OtherThingCollection> pOthers;
    pOthers.getByLabel(events,"OtherThing","testUserTag");
    
    if(0==index) {
      CPPUNIT_ASSERT(not pOthers.isValid());
      CPPUNIT_ASSERT(pOthers.failedToGet());
      CPPUNIT_ASSERT_THROW(pOthers.product(), cms::Exception);
    } else {
      CPPUNIT_ASSERT(pOthers.isValid());
    }
    
    edm::Handle<edmtest::OtherThingCollection> edmPOthers;
    events.getByLabel(tag, edmPOthers);
    if(0==index) {
      CPPUNIT_ASSERT(not edmPOthers.isValid());
      CPPUNIT_ASSERT(edmPOthers.failedToGet());
      CPPUNIT_ASSERT_THROW(edmPOthers.product(), cms::Exception);
    } else {
      CPPUNIT_ASSERT(edmPOthers.isValid());
    }
    
    
  }
}

void testRefInROOT::testHandleErrors()
{
   fwlite::Handle<edmtest::ThingCollection> pThings ;
   CPPUNIT_ASSERT_THROW(*pThings,cms::Exception);
   
   //try copy constructor
   fwlite::Handle<edmtest::ThingCollection> pThings2(pThings) ;
   CPPUNIT_ASSERT_THROW(*pThings2,cms::Exception);
   
}

void testRefInROOT::testTwoGoodFiles()
{
  /*
  std::cout <<"gFile "<<gFile<<std::endl;
  TFile file("good.root");
  std::cout <<" file :" << &file<<" gFile: "<<gFile<<std::endl;
  
  TTree* events = dynamic_cast<TTree*>(file.Get(edm::poolNames::eventTreeName()));
  
  testTree(events);
  std::cout <<"working on second file"<<std::endl;
  TFile file2("good2.root");
  std::cout <<" file2 :"<< &file2<<" gFile: "<<gFile<<std::endl;
  events = dynamic_cast<TTree*>(file2.Get(edm::poolNames::eventTreeName()));
  
  testTree(events);
   */
}


void testRefInROOT::testGoodChain()
{
  /*
  TChain eventChain(edm::poolNames::eventTreeName());
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
  TChain eventChain(edm::poolNames::eventTreeName());
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

void testRefInROOT::testThinning() {

  std::vector<std::string> files { "good.root", "good.root" };
  fwlite::ChainEvent events(files);

  for(events.toBegin(); not events.atEnd(); ++events) {

    fwlite::Handle<std::vector<edmtest::TrackOfThings> > pTrackOfThingsDPlus;
    pTrackOfThingsDPlus.getByLabel(events,"trackOfThingsProducerDPlus");

    fwlite::Handle<std::vector<edmtest::TrackOfThings> > pTrackOfThingsG;
    pTrackOfThingsG.getByLabel(events,"trackOfThingsProducerG");

    fwlite::Handle<std::vector<edmtest::TrackOfThings> > pTrackOfThingsM;
    pTrackOfThingsM.getByLabel(events,"trackOfThingsProducerM");

    // The values in the tests below have no particular meaning.
    // It is just checking that we read the values known to be
    // be put in by the relevant producer.

    int offset = static_cast<int>(100 + 100 * events.eventAuxiliary().event());

    // In the D branch this tests accessing a value in
    // thinned collection made from a thinned collection
    // made from a master collection.
    edmtest::TrackOfThings const& trackD = pTrackOfThingsDPlus->at(0);
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
    edmtest::TrackOfThings const& trackG = pTrackOfThingsG->at(0);
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
    edmtest::TrackOfThings const& trackM0 = pTrackOfThingsM->at(0);
    CPPUNIT_ASSERT(!trackM0.ref1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.ref1.operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM0.ptr1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.ptr1.operator->(), cms::Exception);
    CPPUNIT_ASSERT(!trackM0.refToBase1.isAvailable());
    CPPUNIT_ASSERT_THROW(trackM0.refToBase1.operator->(), cms::Exception);

    edmtest::TrackOfThings const& trackM1 = pTrackOfThingsM->at(1);
    CPPUNIT_ASSERT(trackM1.ref1.isAvailable());
    CPPUNIT_ASSERT(trackM1.ref1->a == 44 + offset);
    CPPUNIT_ASSERT(trackM1.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackM1.ptr1->a == 46 + offset);
    CPPUNIT_ASSERT(trackM1.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackM1.refToBase1->a == 44 + offset);

    edmtest::TrackOfThings const& trackM = pTrackOfThingsM->at(0);
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

  std::vector<std::string> files1 { "refTestCopyDrop.root" };
  std::vector<std::string> files2 { "good.root" };

  fwlite::MultiChainEvent multiChainEvents(files1, files2);
  for (multiChainEvents.toBegin(); ! multiChainEvents.atEnd(); ++multiChainEvents) {

    fwlite::Handle<std::vector<edmtest::TrackOfThings> > pTrackOfThingsDPlus;
    pTrackOfThingsDPlus.getByLabel(multiChainEvents,"trackOfThingsProducerDPlus");

    fwlite::Handle<std::vector<edmtest::TrackOfThings> > pTrackOfThingsG;
    pTrackOfThingsG.getByLabel(multiChainEvents,"trackOfThingsProducerG");

    // The values in the tests below have no particular meaning.
    // It is just checking that we read the values known to be
    // be put in by the relevant producer.

    int offset = static_cast<int>(100 + 100 * multiChainEvents.eventAuxiliary().event());

    // In the D branch this tests accessing a value in
    // thinned collection made from a thinned collection
    // made from a master collection.
    edmtest::TrackOfThings const& trackD = pTrackOfThingsDPlus->at(0);
    CPPUNIT_ASSERT(trackD.ref1.isAvailable());
    CPPUNIT_ASSERT(trackD.ref1->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackD.ptr1->a == 12 + offset);
    CPPUNIT_ASSERT(trackD.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackD.refToBase1->a == 10 + offset);

    CPPUNIT_ASSERT(trackD.ptrVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptrVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.ptrVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackD.ptrVector1[9]->a == 21 + offset);
    CPPUNIT_ASSERT(!trackD.ptrVector1.isAvailable());
    CPPUNIT_ASSERT(trackD.ptrVector1[0]->a == 10 + offset);
    CPPUNIT_ASSERT(trackD.ptrVector1[4]->a == 14 + offset);
    CPPUNIT_ASSERT_THROW(trackD.ptrVector1[8].operator->(), cms::Exception);
    CPPUNIT_ASSERT(trackD.ptrVector1[9]->a == 21 + offset);

    // In the G branch this tests accessing a value in
    // thinned collection made from a master collection.
    // Otherwise the tests are very similar the preceding
    // tests.
    edmtest::TrackOfThings const& trackG = pTrackOfThingsG->at(0);
    CPPUNIT_ASSERT(trackG.ref1.isAvailable());
    CPPUNIT_ASSERT(trackG.ref1->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptr1.isAvailable());
    CPPUNIT_ASSERT(trackG.ptr1->a == 22 + offset);
    CPPUNIT_ASSERT(trackG.refToBase1.isAvailable());
    CPPUNIT_ASSERT(trackG.refToBase1->a == 20 + offset);

    CPPUNIT_ASSERT(trackG.ptrVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[8]->a == 28 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1.isAvailable());
    CPPUNIT_ASSERT(trackG.ptrVector1[0]->a == 20 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[4]->a == 24 + offset);
    CPPUNIT_ASSERT(trackG.ptrVector1[8]->a == 28 + offset);
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

