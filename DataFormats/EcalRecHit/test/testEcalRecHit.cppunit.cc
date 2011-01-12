/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

class testEcalRecHit: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEcalRecHit);
  CPPUNIT_TEST(testOne);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void testOne();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalRecHit);

void testEcalRecHit::testOne(){
  

  // test flagbit setter
  EcalRecHit rh ;
  rh.setFlag(EcalRecHit::kTPSaturated);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated ) );
  
  // test chi2
  float chi2=1.8;
  rh.setChi2(chi2);  
  
  std::cout << " chi2 " << rh.chi2() << std::endl;
  //CPPUNIT_ASSERT(rh.chi2() == chi2);
   
  // test timerr
  float timerr = 0.02;
  EcalUncalibratedRecHit urh;
  urh.setJitterError(timerr);
  rh.setTimeError(urh.jitterErrorBits());
  std::cout << " timerr " << timerr << " " <<rh.timeError() <<  " " <<urh.jitterError() <<std::endl;
  
  std::cout << " **** WARNING : SOMETHING WRONG MIGHT BE GOING ON HERE "<<std::endl;

  //CPPUNIT_ASSERT(rh.timeError() == timerr);
  //CPPUNIT_ASSERT(urh.jitterError()== rh.timeError());

  // did we modify some bit by mistake ?
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated) );
}
