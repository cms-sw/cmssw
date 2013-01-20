/*
 *  serviceregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */
#include <iostream>

#include "FWCore/ServiceRegistry/interface/connect_but_block_self.h"

#include <cppunit/extensions/HelperMacros.h>

class testConnectButBlockSelf: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testConnectButBlockSelf);
   
   CPPUNIT_TEST(test);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testConnectButBlockSelf);

namespace {
   struct ReSignaller {
      edm::signalslot::Signal<void()>* signal_;
      int& seen_;

      ReSignaller(edm::signalslot::Signal<void()>&iSig, int& iSee): signal_(&iSig), seen_(iSee){}
      
      void operator()() {
         //std::cout <<"see signal"<<std::endl;
         ++seen_;
         (*signal_)();
      }
   };
}

void
testConnectButBlockSelf::test()
{
   using namespace edm::serviceregistry;
   edm::signalslot::Signal<void()> theSignal;
   
   int iOne(0);
   ReSignaller one(theSignal,iOne);
   connect_but_block_self(theSignal, one );
   
   int iTwo(0);
   ReSignaller two(theSignal,iTwo);
   connect_but_block_self(theSignal, two );
   
   theSignal();
   
   //std::cout <<"one "<<iOne <<std::endl;
   //std::cout <<"two "<<iTwo <<std::endl;
   
   //Both see two changes, but they never see their 'own' change
   CPPUNIT_ASSERT(iOne == 2);
   CPPUNIT_ASSERT(iTwo == 2);
}
