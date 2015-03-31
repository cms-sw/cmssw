/*
 *  recordwriter.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 09/12/09.
 *
 */
#include <vector>
#include <iostream>

#include "TFile.h"

#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "DataFormats/FWLite/interface/EventSetup.h"
#include "DataFormats/FWLite/interface/ESHandle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

class testRecord: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testRecord);
   
   CPPUNIT_TEST(testGood);
   CPPUNIT_TEST(testFailures);
   
   CPPUNIT_TEST_SUITE_END();
   static bool s_firstSetup;
public:
   void setUp();
   void tearDown(){}
   
   void testGood();
   void testFailures();
};


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRecord);

bool testRecord::s_firstSetup =true;

void testRecord::setUp()
{ 
   if( s_firstSetup) {
      s_firstSetup = false;
      
      //create the test file
      TFile f("testRecord.root","RECREATE");
      fwlite::RecordWriter w("TestRecord", &f);
      
      std::vector<int> v;
      edmtest::SimpleDerived d;
      for(int index=0; index<6; ++index) {
         //std::cout <<" index "<<index<<std::endl;
         v.push_back(index);
         w.update(&v, typeid(v),"");

         d.key = index;
         d.value = index;
         d.dummy = index;
         w.update(&d, typeid(edmtest::Simple),"");

         w.fill(edm::ESRecordAuxiliary(edm::EventID(index*2+1,0,0),edm::Timestamp()));
      }
      w.write();
      f.Close();
      
   } 
}

void testRecord::testGood()
{
   {
      TFile f("testRecord.root","READ");
      
      fwlite::EventSetup es(&f);
      
      std::vector<std::string> recordNames = es.namesOfAvailableRecords();
      CPPUNIT_ASSERT(recordNames.size() == 1);
      CPPUNIT_ASSERT(recordNames[0] == "TestRecord");
      
      
      fwlite::RecordID testRecID = es.recordID("TestRecord");

      std::vector<std::pair<std::string,std::string> > dataIds = 
      es.get(testRecID).typeAndLabelOfAvailableData();
      
      CPPUNIT_ASSERT(dataIds.size() == 2);
      unsigned int matches =0;
      for(std::vector<std::pair<std::string,std::string> >::const_iterator it = dataIds.begin(),
          itEnd = dataIds.end();
          it != itEnd;
          ++it) {
         std::cout <<it->first<< " '"<<it->second<<"'"<<std::endl;
         if( (it->first == "std::vector<int>") &&
            (it->second =="") ) {
            ++matches;
            continue;
         }
         if( (it->first == "edmtest::Simple") &&
            (it->second =="") ) {
            ++matches;
         }
      }
      
      CPPUNIT_ASSERT(2==matches);
      
      for(unsigned int index=1; index<10; ++index) {
         es.syncTo(edm::EventID(index,0,0),edm::Timestamp());
         unsigned int run = index;
         if(0!=(run-1)%2) {
            --run;
         }
         
         fwlite::ESHandle<std::vector<int> > vIntHandle;
         CPPUNIT_ASSERT(es.get(testRecID).get(vIntHandle));
         CPPUNIT_ASSERT(vIntHandle.isValid());
         //std::cout <<" index "<<index<<" size "<<vIntHandle->size()<<" "<<es.get(testRecID).startSyncValue().eventID()<<std::endl;
         CPPUNIT_ASSERT(es.get(testRecID).startSyncValue().eventID().run() == run);
         CPPUNIT_ASSERT(vIntHandle->size()==(index-1)/2+1);
         
         fwlite::ESHandle<edmtest::Simple> simpleHandle;
         CPPUNIT_ASSERT(es.get(testRecID).get(simpleHandle));
         CPPUNIT_ASSERT(simpleHandle->key == static_cast<int>((index-1)/2));
      }      
   }
}

struct DummyWithNoDictionary {};

void testRecord::testFailures()
{
   TFile f("testRecord.root","READ");
   
   fwlite::EventSetup es(&f);
   
   CPPUNIT_ASSERT(not es.exists("DoesNotExist"));
   CPPUNIT_ASSERT_THROW(es.recordID("DoesNotExist"),cms::Exception);

   fwlite::RecordID testRecID = es.recordID("TestRecord");

   const fwlite::Record& testRecord = es.get(testRecID);

   fwlite::ESHandle<std::vector<int> > vIntHandle;
   CPPUNIT_ASSERT(not vIntHandle.isValid());
   
   CPPUNIT_ASSERT(not testRecord.get(vIntHandle));
   CPPUNIT_ASSERT(not vIntHandle.isValid());
   CPPUNIT_ASSERT_THROW(*vIntHandle, cms::Exception);
   
   es.syncTo(edm::EventID(1,0,0),edm::Timestamp());
   
   CPPUNIT_ASSERT(not testRecord.get(vIntHandle, "notThere"));
   
   fwlite::ESHandle<std::vector<DummyWithNoDictionary> > noDictHandle;
   CPPUNIT_ASSERT(not testRecord.get(noDictHandle));
   CPPUNIT_ASSERT_THROW(*noDictHandle,cms::Exception);
   CPPUNIT_ASSERT_THROW(noDictHandle.operator->(),cms::Exception);
   
}

