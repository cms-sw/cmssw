/*
 *  sharedresourcesregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Framework/src/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

using namespace edm;

class testSharedResourcesRegistry: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testSharedResourcesRegistry);
   
   CPPUNIT_TEST(oneTest);
   CPPUNIT_TEST(legacyTest);
   CPPUNIT_TEST(multipleTest);
  
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void oneTest();
   void legacyTest();
   void multipleTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSharedResourcesRegistry);


void testSharedResourcesRegistry::oneTest()
{
  edm::SharedResourcesRegistry reg;
  
  reg.registerSharedResource("foo");
  reg.registerSharedResource("bar");
  reg.registerSharedResource("zoo");

  {
    std::vector<std::string> res{"foo","bar","zoo"};
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(0 == tester.numberOfResources());
  }
  {
    std::vector<std::string> res{"foo"};
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(0 == tester.numberOfResources());
  }
}

void testSharedResourcesRegistry::legacyTest()
{
  std::vector<std::string> res{edm::SharedResourcesRegistry::kLegacyModuleResourceName};
  {
    edm::SharedResourcesRegistry reg;
    
    reg.registerSharedResource(edm::SharedResourcesRegistry::kLegacyModuleResourceName);
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(0 == tester.numberOfResources());
  }
  {
    edm::SharedResourcesRegistry reg;
    
    reg.registerSharedResource(edm::SharedResourcesRegistry::kLegacyModuleResourceName);
    reg.registerSharedResource(edm::SharedResourcesRegistry::kLegacyModuleResourceName);

    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(1 == tester.numberOfResources());
    
  }
}

void testSharedResourcesRegistry::multipleTest()
{
  edm::SharedResourcesRegistry reg;
  
  reg.registerSharedResource("foo");
  reg.registerSharedResource("bar");
  reg.registerSharedResource("zoo");
  reg.registerSharedResource("zoo");
  reg.registerSharedResource("bar");
  reg.registerSharedResource("zoo");
  
  {
    std::vector<std::string> res{"foo","bar","zoo"};
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(2 == tester.numberOfResources());
  }
  {
    std::vector<std::string> res{"foo"};
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(0 == tester.numberOfResources());
  }
  {
    std::vector<std::string> res{"bar"};
    auto tester = reg.createAcquirer(res);
    
    CPPUNIT_ASSERT(1 == tester.numberOfResources());
  }

}
