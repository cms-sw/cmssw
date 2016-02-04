/**
   \file
   test file for EcalShowerContainmentCorrections 

   \author Stefano ARGIRO
   \version $Id: testEcalShowerContaimentCorrections.cc,v 1.1 2007/05/15 20:37:21 argiro Exp $
   \date 28 Jun 2005
*/

#include <cppunit/extensions/HelperMacros.h>
#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
class testEcalShowerContainmentCorrections : public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testEcalShowerContainmentCorrections);

  CPPUNIT_TEST(testFillandReadBack1);
  CPPUNIT_TEST(testFillandReadBack2);
 
  CPPUNIT_TEST_SUITE_END();

public:


  void setUp();
  void tearDown();  
  void testFillandReadBack1();
  void testFillandReadBack2(); 
 
  double * testdata_;
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalShowerContainmentCorrections);

void testEcalShowerContainmentCorrections::setUp(){

  testdata_ = new double[EcalShowerContainmentCorrections::Coefficients::kSize];

  for (unsigned int i=0; i<EcalShowerContainmentCorrections::Coefficients::kSize;
       ++i) testdata_[i]=i;
}

void testEcalShowerContainmentCorrections::tearDown(){
  delete[] testdata_;
}

void testEcalShowerContainmentCorrections::testFillandReadBack1(){
  
  EcalShowerContainmentCorrections::Coefficients coeff;
  std::copy(testdata_, 
	    testdata_+EcalShowerContainmentCorrections::Coefficients::kSize,
	    coeff.data);
  
  // fill a random xtal
  EBDetId xtal(5,550,EBDetId::SMCRYSTALMODE);

  EcalShowerContainmentCorrections correction;
  
  correction.fillCorrectionCoefficients(xtal,3,coeff);


  
  EcalShowerContainmentCorrections::Coefficients correction_readback=
    correction.correctionCoefficients(xtal);
  
  for (unsigned int i=0; i<EcalShowerContainmentCorrections::Coefficients::kSize;
       ++i) 
    CPPUNIT_ASSERT(testdata_[i]==correction_readback.data[i]);  

}


void testEcalShowerContainmentCorrections::testFillandReadBack2(){
  
  EcalShowerContainmentCorrections::Coefficients coeff;
  std::copy(testdata_, 
	    testdata_+EcalShowerContainmentCorrections::Coefficients::kSize,
	    coeff.data);
  
 
  EcalShowerContainmentCorrections correction;

  // use filling by module
  correction.fillCorrectionCoefficients(3,2,coeff);

  // this is a xtal in SM3 , module 2
  EBDetId xtal(3, 550, EBDetId::SMCRYSTALMODE);

  EcalShowerContainmentCorrections::Coefficients correction_readback=
    correction.correctionCoefficients(xtal);
  
  for (unsigned int i=0; i<EcalShowerContainmentCorrections::Coefficients::kSize;
       ++i) 
    CPPUNIT_ASSERT(testdata_[i]==correction_readback.data[i]);  

}


#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
