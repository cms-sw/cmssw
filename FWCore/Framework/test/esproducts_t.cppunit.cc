/*
 *  esproducts_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/18/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Framework/interface/ESProducts.h"
using edm::ESProducts;
using edm::es::products;

class testEsproducts: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testEsproducts);
CPPUNIT_TEST(constPtrTest);
CPPUNIT_TEST(uniquePtrTest);
CPPUNIT_TEST(sharedPtrTest);
CPPUNIT_TEST(manyTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void constPtrTest();
  void uniquePtrTest();
  void sharedPtrTest();
  void manyTest();
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

   product.moveTo(readInt);
   product.moveTo(readFloat);
   
   CPPUNIT_ASSERT(readInt == &int_);
   CPPUNIT_ASSERT(readFloat == &float_);
}

void testEsproducts::uniquePtrTest()
{
  constexpr int kInt = 5;
  auto int_ = std::make_unique<int>(kInt);
  constexpr float kFloat = 3.1;
  auto float_ = std::make_unique<float>(kFloat);
  
  ESProducts<std::unique_ptr<int>, std::unique_ptr<float>> product = products(std::move(int_), std::move(float_));
  
  std::unique_ptr<int> readInt;
  std::unique_ptr<float> readFloat;
  
  product.moveTo(readInt);
  product.moveTo(readFloat);
  
  CPPUNIT_ASSERT(*readInt == kInt);
  CPPUNIT_ASSERT(*readFloat == kFloat);
}

void testEsproducts::sharedPtrTest()
{
  auto int_ = std::make_shared<int>(5);
  auto float_ = std::make_shared<float>(3.1);
  
  ESProducts<std::shared_ptr<int>, std::shared_ptr<float>> product =
  products(int_,float_);
  
  std::shared_ptr<int> readInt;
  std::shared_ptr<float> readFloat;
  
  product.moveTo(readInt);
  product.moveTo(readFloat);
  
  CPPUNIT_ASSERT(readInt.get() == int_.get());
  CPPUNIT_ASSERT(readFloat.get() == float_.get());
}


ESProducts<const int*, const float*, const double*>
returnManyPointers(const int* iInt, const float* iFloat, const double* iDouble)
{
   return edm::es::products(iInt, iFloat,  iDouble);
}

void testEsproducts::manyTest()
{
   int int_ = 0;
   float float_ = 0;
   double double_ = 0;
   
   ESProducts<const int*, const float*, const double*> product = 
      returnManyPointers(&int_, &float_,&double_);
   
   const int* readInt = 0;
   const float* readFloat = 0;
   const double* readDouble = 0;
   
   product.moveTo(readInt);
   product.moveTo(readFloat);
   product.moveTo(readDouble);
   
   CPPUNIT_ASSERT(readInt == &int_);
   CPPUNIT_ASSERT(readFloat == &float_);
   CPPUNIT_ASSERT(readDouble == &double_);
}

   
