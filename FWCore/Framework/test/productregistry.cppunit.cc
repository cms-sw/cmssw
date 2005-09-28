/**
   \file
   test for ProductRegistry 

   \author Stefano ARGIRO
   \version $Id: productregistry.cppunit.cc,v 1.1 2005/07/21 21:07:14 argiro Exp $
   \date 21 July 2005
*/



#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Framework/interface/EventProcessor.h>

class testProductRegistry: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testProductRegistry);

CPPUNIT_TEST(testProductRegistration);

CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testProductRegistration();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testProductRegistry);


void  testProductRegistry:: testProductRegistration(){


  const std::string config=

    "process TEST = { \n"
      "module m1 = TestPRegisterModule1{ } \n"
      "module m2 = TestPRegisterModule2{ } \n" 
      "path p = {m1,m2}\n"
      "source = TestInputSource4ProductRegistry{ }\n"
    "}\n";

   edm::EventProcessor proc(config);
 
}
