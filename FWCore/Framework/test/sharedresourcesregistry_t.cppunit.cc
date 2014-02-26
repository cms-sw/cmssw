/*
 *  sharedresourcesregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#define SHAREDRESOURCETESTACCESSORS 1
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

  CPPUNIT_ASSERT(reg.resourceMap().size() == 0);

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
  {
    edm::SharedResourcesRegistry reg;

    reg.registerSharedResource("foo");
    reg.registerSharedResource("zoo");

    auto const& resourceMap = reg.resourceMap();
    CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).first.get() == nullptr);
    CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).first.get() == nullptr);
    CPPUNIT_ASSERT(resourceMap.size() == 2);
    CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).second == 1);
    CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).second == 1);

    reg.registerSharedResource(edm::SharedResourcesRegistry::kLegacyModuleResourceName);
    CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).first.get() != nullptr);
    CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).first.get() != nullptr);
    CPPUNIT_ASSERT(resourceMap.at(edm::SharedResourcesRegistry::kLegacyModuleResourceName).first.get() != nullptr);
    CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).second == 2);
    CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).second == 2);
    CPPUNIT_ASSERT(resourceMap.at(edm::SharedResourcesRegistry::kLegacyModuleResourceName).second == 1);
    CPPUNIT_ASSERT(resourceMap.size() == 3);

    reg.registerSharedResource(edm::SharedResourcesRegistry::kLegacyModuleResourceName);
    reg.registerSharedResource("bar");
    reg.registerSharedResource("zoo");
    CPPUNIT_ASSERT(resourceMap.at(std::string("bar")).first.get() != nullptr);
    CPPUNIT_ASSERT(resourceMap.at(std::string("bar")).second == 3);
    CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).second == 3);
    CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).second == 4);
    CPPUNIT_ASSERT(resourceMap.at(edm::SharedResourcesRegistry::kLegacyModuleResourceName).second == 2);

    auto tester = reg.createAcquirer(res);

    CPPUNIT_ASSERT(3 == tester.numberOfResources());
  }
}

void testSharedResourcesRegistry::multipleTest()
{
  edm::SharedResourcesRegistry reg;
  auto const& resourceMap = reg.resourceMap();
  
  reg.registerSharedResource("foo");
  reg.registerSharedResource("bar");
  reg.registerSharedResource("zoo");
  reg.registerSharedResource("zoo");
  reg.registerSharedResource("bar");
  reg.registerSharedResource("zoo");

  CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).first.get() == nullptr);
  CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).first.get() != nullptr);
  CPPUNIT_ASSERT(resourceMap.at(std::string("bar")).first.get() != nullptr);
  CPPUNIT_ASSERT(resourceMap.at(std::string("bar")).second == 2);
  CPPUNIT_ASSERT(resourceMap.at(std::string("foo")).second == 1);
  CPPUNIT_ASSERT(resourceMap.at(std::string("zoo")).second == 3);
  CPPUNIT_ASSERT(resourceMap.size() == 3);

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
