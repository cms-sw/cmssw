/*----------------------------------------------------------------------

Test of the EventProcessor class.

$Id: eventprocessor_t.cppunit.cc,v 1.4 2005/07/19 16:34:00 viji Exp $

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cppunit/extensions/HelperMacros.h>

class testeventprocessor: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprocessor);
CPPUNIT_TEST(parseTest);
CPPUNIT_TEST(prepostTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void parseTest();
  void prepostTest();
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

void testeventprocessor::parseTest()
{
  int rc = -1;                // we should never return this value!
  try { work(); rc = 0;}
  catch (seal::Error& e)
    {
      std::cerr << "Application exception caught: "
		<< e.explainSelf() << std::endl;
      CPPUNIT_ASSERT( "Caught seal::Error " == 0 );
    }
  catch (std::exception& e)
    {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
     CPPUNIT_ASSERT( "Caught std::exception " == 0 );
    }
  catch (...)
    {
     CPPUNIT_ASSERT( "Caught unknown exception " == 0 );
    }
}

static int g_pre = 0;
static int g_post = 0;
static
void doPre(const edm::Event&, const edm::EventSetup& ) 
{
   ++g_pre;
}
static
void doPost(const edm::Event&, const edm::EventSetup& ) 
{
   CPPUNIT_ASSERT( g_pre == ++g_post );
}

void testeventprocessor::prepostTest()
{
   std::string configuration("process p = {\n"
                             "source = EmptyInputService { untracked int32 maxEvents = 5 }\n"
                             "module m1 = TestMod { int32 ivalue = -3 }\n"
                             "path p1 = { m1 }\n"
                             "}\n");
   edm::EventProcessor proc(configuration);
   
   proc.preProcessEventSignal.connect( &doPre );
   proc.postProcessEventSignal.connect( &doPost );
   proc.run(0);
   CPPUNIT_ASSERT( 5 == g_pre );
   CPPUNIT_ASSERT( 5 == g_post );
}
