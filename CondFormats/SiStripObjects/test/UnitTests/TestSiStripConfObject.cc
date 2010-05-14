#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#define protected public
#define private public
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#undef protected
#undef private

class TestSiStripConfObject : public CppUnit::TestFixture
{
 public:
  TestSiStripConfObject() {}

  void setUp() {}

  void tearDown() {}

  void testPutAndGet()
  {
    CPPUNIT_ASSERT( conf.put("par1", 1) == true );
    CPPUNIT_ASSERT( conf.put("par2", 1.1) == true );
    CPPUNIT_ASSERT( conf.put("par3", "one") == true);

    CPPUNIT_ASSERT( conf.getInt("par1") == 1 );
    CPPUNIT_ASSERT( float(conf.getDouble("par2")) == float(1.1) );
    CPPUNIT_ASSERT( conf.getString("par3") == "one" );

    // std::cout << "par1 int = " << conf.getInt("par1") << std::endl;
    // std::cout << "par1 float = " << conf.getFloat("par1") << std::endl;
    // std::cout << "par1 string = " << conf.getString("par1") << std::endl;

    // std::cout << "par2 int = " << conf.getInt("par2") << std::endl;
    // std::cout << "par2 float = " << conf.getFloat("par2") << std::endl;
    // std::cout << "par2 string = " << conf.getString("par2") << std::endl;

    // std::cout << "par3 int = " << conf.getInt("par3") << std::endl;
    // std::cout << "par3 float = " << conf.getFloat("par3") << std::endl;
    // std::cout << "par3 string = " << conf.getString("par3") << std::endl;

    // Insertion of already existing parameter
    CPPUNIT_ASSERT( conf.put("par1", 2) == false );

    // Retrieval of inexistent parameter
    CPPUNIT_ASSERT( conf.getInt("par4") == 0 );
    CPPUNIT_ASSERT( conf.getDouble("par4") == 0. );
    CPPUNIT_ASSERT( conf.getString("par4") == "" );
  }

  CPPUNIT_TEST_SUITE( TestSiStripConfObject );
  CPPUNIT_TEST( testPutAndGet );
  CPPUNIT_TEST_SUITE_END();

  SiStripConfObject conf;
};

CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripConfObject );
