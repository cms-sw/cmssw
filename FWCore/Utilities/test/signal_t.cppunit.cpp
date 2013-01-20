/*----------------------------------------------------------------------

Test program for edm::signalslot::Signal class.

$Id$
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/Signal.h"

class testSignal: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testSignal);

CPPUNIT_TEST(connectTest);
CPPUNIT_TEST(emitTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void connectTest();
  void emitTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSignal);

static int s_value = 0;
static void setValueFunct(int iValue) { s_value = iValue;}

void testSignal::connectTest()

{
   edm::signalslot::Signal<void(int)> sig;
   CPPUNIT_ASSERT(sig.slots().size()==0);
   
   int value1 = 0;
   sig.connect([&](int iValue)->void{ value1=iValue;});
   
   CPPUNIT_ASSERT(sig.slots().size()==1);

   int value2=0;
   sig.connect([&](int iValue){value2=iValue;});
   CPPUNIT_ASSERT(sig.slots().size()==2);

   sig.connect(setValueFunct);
   //see that the slots we created are actually there
   for(auto const& slot: sig.slots()) {
      slot(5);
   }
   CPPUNIT_ASSERT(value1==5);
   CPPUNIT_ASSERT(value2==5);
   CPPUNIT_ASSERT(value2==s_value);
}

void testSignal::emitTest()
{
   edm::signalslot::Signal<void(int)> sig;
   
   int value1 = 0;
   sig.connect([&](int iValue){ value1=iValue;});
   
   int value2=0;
   sig.connect([&](int iValue){value2=iValue;});
   
   sig.emit(5);
   CPPUNIT_ASSERT(value1==5);
   CPPUNIT_ASSERT(value2==5);

   sig.emit(1);
   CPPUNIT_ASSERT(value1==1);
   CPPUNIT_ASSERT(value2==1);
}
