/*----------------------------------------------------------------------

Test of the EventProcessor class.

$Id: EventProcessor_t.cpp,v 1.4 2005/06/28 04:46:02 jbk Exp $

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>

#include "FWCore/CoreFramework/interface/EventProcessor.h"
#include "FWCore/FWUtilities/interface/Exception.h"

#include <cppunit/extensions/HelperMacros.h>

class testeventprocessor: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprocessor);
CPPUNIT_TEST(eventprocessorTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void eventprocessorTest();
private:
void work()
{
  std::string configuration("process p = {\n"
			    "source = EmptyInputService { untracked int32 maxEvents = 5 }\n"
			    "module m1 = TestMod { int32 ivalue = 10 }\n"
			    "module m2 = TestMod { int32 ivalue = -3 }\n"
                            "path p1 = { m1,m2 }\n"
			    "}\n");
  edm::EventProcessor proc(configuration);

  proc.run(0);
}
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprocessor);

void testeventprocessor::eventprocessorTest()
//int main()
{
  int rc = -1;                // we should never return this value!
  try { work(); rc = 0;}
  catch (seal::Error& e)
    {
      std::cerr << "Application exception caught: "
		<< e.explainSelf() << std::endl;
      rc = 1;
    }
  catch (std::exception& e)
    {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
      rc = 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception caught" << std::endl;
      rc = 2;
    }
//  return rc;
}
