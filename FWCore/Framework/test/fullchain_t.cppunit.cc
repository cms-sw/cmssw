/*
 *  full_chain_test.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/3/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/Timestamp.h"
#include "FWCore/CoreFramework/interface/DataProxyProvider.h"
#include "FWCore/CoreFramework/interface/HCMethods.icc"
#include "FWCore/CoreFramework/interface/recordGetImplementation.icc"
#include "FWCore/CoreFramework/interface/ESHandle.h"
#include "FWCore/CoreFramework/interface/DataProxyTemplate.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/CoreFramework/test/DummyRecord.h"
#include "FWCore/CoreFramework/test/DummyData.h"
#include "FWCore/CoreFramework/test/DummyFinder.h"
#include "FWCore/CoreFramework/test/DummyProxyProvider.h"
#include <cppunit/extensions/HelperMacros.h>

using namespace edm;
using namespace edm::eventsetup;
using namespace edm::eventsetup::test;

class testfullChain: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testfullChain);

CPPUNIT_TEST(getfromDataproxyproviderTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void getfromDataproxyproviderTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfullChain);

void testfullChain::getfromDataproxyproviderTest()
{
   eventsetup::EventSetupProvider provider;

   boost::shared_ptr<DataProxyProvider> pProxyProv(new DummyProxyProvider);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));

   pFinder->setInterval(ValidityInterval(1,5));
   for(unsigned int iTime=1; iTime != 6; ++iTime) {
      EventSetup const& eventSetup = provider.eventSetupForInstance(Timestamp(iTime));
      ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != &(*pDummy));
      
      eventSetup.getData(pDummy);
   
      CPPUNIT_ASSERT(0 != &(*pDummy));
   }
}
