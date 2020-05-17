/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

class testEcalRecHit : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEcalRecHit);
  CPPUNIT_TEST(testOne);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testOne();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalRecHit);

void testEcalRecHit::testOne() {
  // test flagbit setter
  EcalRecHit rh(EBDetId(1, 1), 0., 0.);
  rh.setFlag(EcalRecHit::kTPSaturated);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated));

  rh.setFlag(EcalRecHit::kHasSwitchToGain6);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kHasSwitchToGain6));

  rh.setFlag(EcalRecHit::kHasSwitchToGain1);
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kHasSwitchToGain1));
  rh.unsetFlag(EcalRecHit::kHasSwitchToGain1);
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kHasSwitchToGain1));

  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kSaturated));

  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kPoorReco));
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kUnknown));
  CPPUNIT_ASSERT(!rh.checkFlag(EcalRecHit::kGood));

  // did we modify some bit by mistake ?
  CPPUNIT_ASSERT(rh.checkFlag(EcalRecHit::kTPSaturated));

  // check unsetting of flag
  EcalRecHit rh2(EBDetId(1, 1), 0., 0.);
  rh2.setFlag(EcalRecHit::kOutOfTime);
  CPPUNIT_ASSERT(rh2.checkFlag(EcalRecHit::kOutOfTime));
  rh2.unsetFlag(EcalRecHit::kOutOfTime);
  CPPUNIT_ASSERT(!rh2.checkFlag(EcalRecHit::kOutOfTime));

  rh2.unsetFlag(EcalRecHit::kGood);
  CPPUNIT_ASSERT(!rh2.checkFlag(EcalRecHit::kGood));

  // test packing of values
  CPPUNIT_ASSERT_EQUAL((uint32_t)0, EcalRecHit::getMasked(0, 8, 16));
  // setting value of 12 at offset of 8; pack width=16
  CPPUNIT_ASSERT_EQUAL((uint32_t)3072, EcalRecHit::setMasked(0, 12, 8, 16));
  CPPUNIT_ASSERT_EQUAL((uint32_t)12, EcalRecHit::getMasked(3072, 8, 16));
  // check if setting clears the old value
  CPPUNIT_ASSERT_EQUAL((uint32_t)(3072 + 0xff), EcalRecHit::setMasked(0x00aaffff, 12, 8, 16));
  CPPUNIT_ASSERT_EQUAL((uint32_t)12, EcalRecHit::getMasked(3072 + 0xff, 8, 16));

  // test chi2
  rh.setChi2(1.8);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.8, rh.chi2(), 0.5);
  rh.setChi2(50.8);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(50.8, rh.chi2(), 0.5);

  // test timeError
  float timerr = 0.12;
  EcalUncalibratedRecHit urh;
  urh.setJitterError(timerr);
  rh.setTimeError(urh.jitterErrorBits());

  CPPUNIT_ASSERT_DOUBLES_EQUAL(timerr * 25, rh.timeError(), 0.5);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(urh.jitterError() * 25, rh.timeError(), 0.00001);

  // test energyError
  for (float x = 0.0011; x < 10.e4; x *= 5.) {
    rh.setEnergyError(x);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(x, rh.energyError(), 0.01 * x);
  }
}
