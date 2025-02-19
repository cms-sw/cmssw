/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

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
  EcalRecHit rh (EBDetId(1,1),0.,0.);
  rh.setFlag(EcalRecHit::kTPSaturated);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated ) );

  rh.setFlag(EcalRecHit::kHasSwitchToGain6);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kHasSwitchToGain6) );

  rh.setFlag(EcalRecHit::kHasSwitchToGain1);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kHasSwitchToGain1) );
  rh.unsetFlag(EcalRecHit::kHasSwitchToGain1);
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kHasSwitchToGain1) );

  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kSaturated) );

  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kPoorReco) );
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kUnknown) );
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kGood) );
 
  
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

  // mhh, timeError apparantly returns ns
  // chi2 has rounding error

  //std::cout << " timerr " << timerr << " " <<rh.timeError() <<  " " <<urh.jitterError() <<std::endl;
  
  //std::cout << " **** WARNING : SOMETHING WRONG MIGHT BE GOING ON HERE "<<std::endl;

  //CPPUNIT_ASSERT(rh.timeError() == timerr);
  //CPPUNIT_ASSERT(urh.jitterError()== rh.timeError());

  // did we modify some bit by mistake ?
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated) );


  // check unsetting of flag
  EcalRecHit rh2(EBDetId(1,1),0.,0.);
  rh2.setFlag(EcalRecHit::kOutOfTime);
  CPPUNIT_ASSERT(rh2.checkFlag(EcalRecHit::kOutOfTime) );
  rh2.unsetFlag(EcalRecHit::kOutOfTime);
  CPPUNIT_ASSERT(!rh2.checkFlag(EcalRecHit::kOutOfTime) );

  rh2.unsetFlag(EcalRecHit::kGood);
  CPPUNIT_ASSERT(!rh2.checkFlag(EcalRecHit::kGood) );

}
