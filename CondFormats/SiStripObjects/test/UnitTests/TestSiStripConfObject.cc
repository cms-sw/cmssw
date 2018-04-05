#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

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

    CPPUNIT_ASSERT( conf.get<int>("par1") == 1 );
    CPPUNIT_ASSERT( conf.get<float>("par2") == float(1.1) );
    CPPUNIT_ASSERT( conf.get<std::string>("par3") == "one" );

    // std::cout << "par1 int = " << conf.get<int>("par1") << std::endl;
    // std::cout << "par1 float = " << conf.get<float>("par1") << std::endl;
    // std::cout << "par1 string = " << conf.get<std::string>("par1") << std::endl;

    // std::cout << "par2 int = " << conf.get<int>("par2") << std::endl;
    // std::cout << "par2 float = " << conf.get<float>("par2") << std::endl;
    // std::cout << "par2 string = " << conf.get<std::string>("par2") << std::endl;

    // std::cout << "par3 int = " << conf.get<int>("par3") << std::endl;
    // std::cout << "par3 float = " << conf.get<float>("par3") << std::endl;
    // std::cout << "par3 string = " << conf.get<std::string>("par3") << std::endl;

    // Insertion of already existing parameter
    CPPUNIT_ASSERT( conf.put("par1", 2) == false );

    // Check if the parameter is there
    CPPUNIT_ASSERT( conf.isParameter("par1") );
    CPPUNIT_ASSERT( conf.isParameter("par2") );
    CPPUNIT_ASSERT( conf.isParameter("par3") );
    CPPUNIT_ASSERT( !(conf.isParameter("par4")) );
  }

  CPPUNIT_TEST_SUITE( TestSiStripConfObject );
  CPPUNIT_TEST( testPutAndGet );
  CPPUNIT_TEST_SUITE_END();

  SiStripConfObject conf;
};

CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripConfObject );
