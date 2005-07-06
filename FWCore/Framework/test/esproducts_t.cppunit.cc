/*
 *  esproducts_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/18/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/CoreFramework/interface/ESProducts.h"
using namespace edm::eventsetup::produce;

class testEsproducts: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testEsproducts);
CPPUNIT_TEST(constPtrTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void constPtrTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEsproducts);


ESProducts<const int*, const float*>
returnPointers(const int* iInt, const float* iFloat)
{
   return products(iInt, iFloat);
}

void testEsproducts::constPtrTest()
{
   int int_ = 0;
   float float_ = 0;
   
   ESProducts<const int*, const float*> product = 
      returnPointers(&int_, &float_);
   
   const int* readInt = 0;
   const float* readFloat = 0;

   product.assignTo(readInt);
   product.assignTo(readFloat);
   
   CPPUNIT_ASSERT(readInt == &int_);
   CPPUNIT_ASSERT(readFloat == &float_);
}

   
