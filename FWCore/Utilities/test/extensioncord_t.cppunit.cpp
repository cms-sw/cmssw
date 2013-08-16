/*----------------------------------------------------------------------

Test program for edm::ExtensionCord class.
Created by Chris Jones on 22-09-2006

 ----------------------------------------------------------------------*/

#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/ExtensionCord.h"
#include "FWCore/Utilities/interface/SimpleOutlet.h"

class testExtensionCord: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testExtensionCord);
  
  CPPUNIT_TEST(unpluggedTest);
  CPPUNIT_TEST(pluggedTest);
  CPPUNIT_TEST(copyTest);
  
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void unpluggedTest();
  void pluggedTest();
  void copyTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testExtensionCord);


void testExtensionCord::unpluggedTest()
{
  edm::ExtensionCord<int> dangling;
  CPPUNIT_ASSERT(!dangling.connected());

  CPPUNIT_ASSERT_THROW(*dangling, cms::Exception);
  CPPUNIT_ASSERT_THROW(dangling.operator->(), cms::Exception);
}

void testExtensionCord::pluggedTest()
{
  edm::ExtensionCord<int> cord;
  
  {
    int value(1);
    edm::ValueHolderECGetter<int> getter(value);
    
    edm::SimpleOutlet<int> outlet( getter, cord );
    
    CPPUNIT_ASSERT( 1 == *cord);
  }
  CPPUNIT_ASSERT(!cord.connected());
}

void testExtensionCord::copyTest()
{
  edm::ExtensionCord<int> cord1;
  edm::ExtensionCord<int> cord2(cord1);
  
  {
    int value(1);
    edm::ValueHolderECGetter<int> getter(value);
    
    edm::SimpleOutlet<int> outlet(getter, cord1 );
    
    CPPUNIT_ASSERT( 1 == *cord2);
  }
  CPPUNIT_ASSERT(!cord2.connected());
}
