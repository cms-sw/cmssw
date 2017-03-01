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
#include "TTree.h"

#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/Provenance/interface/ESRecordAuxiliary.h"
#include "DataFormats/FWLite/interface/format_type_name.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

class testRecordWriter: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testRecordWriter);
   
   CPPUNIT_TEST(testNoInheritance);
   CPPUNIT_TEST(testInheritance);
   
   CPPUNIT_TEST_SUITE_END();
public:
   void setUp(){ }
   void tearDown(){}
   
   void testNoInheritance();
   void testInheritance();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRecordWriter);


void testRecordWriter::testNoInheritance()
{
   {
      TFile f("testRecordWriter.root","RECREATE");
      fwlite::RecordWriter w("TestRecord", &f);
      
      std::vector<int> v;
      for(int index=0; index<10; ++index) {
         //std::cout <<" index "<<index<<std::endl;
         v.push_back(index);
         w.update(&v, typeid(v),"");
         w.fill(edm::ESRecordAuxiliary(edm::EventID(index+1,0,0),edm::Timestamp()));
      }
      w.write();
      f.Close();
   }
   
   {
      TFile f("testRecordWriter.root","READ");
      TTree* recordTree = reinterpret_cast<TTree*>(f.Get("TestRecord"));
      CPPUNIT_ASSERT(0!=recordTree);
      
      edm::ESRecordAuxiliary* aux=0;
      recordTree->SetBranchAddress("ESRecordAuxiliary",&aux);

      std::vector<int>* pV=0;
      recordTree->SetBranchAddress((fwlite::format_type_to_mangled("std::vector<int>")+"__").c_str(),&pV);

      for(int index=0; index < recordTree->GetEntries(); ++index) {
         recordTree->GetEntry(index);
         CPPUNIT_ASSERT( aux->eventID().run()==static_cast<unsigned int>(index+1));
         CPPUNIT_ASSERT( pV->size()==static_cast<size_t>(index+1));
      }
      
   }
}

void testRecordWriter::testInheritance()
{
   {
      TFile f("testRecordWriter.root","RECREATE");
      fwlite::RecordWriter w("TestRecord", &f);
      
      edmtest::SimpleDerived d;
      for(int index=0; index<10; ++index) {
         //std::cout <<" index "<<index<<std::endl;
         d.key = index;
         d.value = index;
         d.dummy = index;
         w.update(&d, typeid(edmtest::Simple),"");
         w.fill(edm::ESRecordAuxiliary(edm::EventID(index+1,0,0),edm::Timestamp()));
      }
      w.write();
      f.Close();
   }
   
   {
      TFile f("testRecordWriter.root","READ");
      TTree* recordTree = reinterpret_cast<TTree*>(f.Get("TestRecord"));
      CPPUNIT_ASSERT(0!=recordTree);
      
      edm::ESRecordAuxiliary* aux=0;
      recordTree->SetBranchAddress("ESRecordAuxiliary",&aux);

      edmtest::Simple* pS=0;
      recordTree->SetBranchAddress((fwlite::format_type_to_mangled("edmtest::Simple")+"__").c_str(),&pS);

      for(int index=0; index < recordTree->GetEntries(); ++index) {
         recordTree->GetEntry(index);
         CPPUNIT_ASSERT( aux->eventID().run()==static_cast<unsigned int>(index+1));
         CPPUNIT_ASSERT( pS->key==index);
         CPPUNIT_ASSERT(0 != dynamic_cast<edmtest::SimpleDerived*>(pS));
         CPPUNIT_ASSERT(index == dynamic_cast<edmtest::SimpleDerived*>(pS)->dummy);
      }      
   }
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
