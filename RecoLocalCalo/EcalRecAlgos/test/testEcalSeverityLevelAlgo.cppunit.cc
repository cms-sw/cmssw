/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class testEcalSeverityLevelAlgo: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEcalSeverityLevelAlgo);
  CPPUNIT_TEST(testSeverity);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown(){delete algo_;}

  void testSeverity();

  EcalSeverityLevelAlgo* algo_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalSeverityLevelAlgo);

void testEcalSeverityLevelAlgo::setUp(){

  edm::ParameterSet ps;
  std::vector<uint32_t> fMask;
  fMask.push_back(0x0001); // good->good
  fMask.push_back(0x0022); // poorreco,poorcalib->problematic
  fMask.push_back(0x0180); // LERecovered,TowRecovered->recovered
  fMask.push_back(0x0004); // outoftime->time 
  fMask.push_back(0x6000); // weird,diweird->weird
  fMask.push_back(0x0698); // faultyhw,noisy,saturated,dead,killed,->bad

  std::vector<uint32_t> dbMask;
  dbMask.push_back(0x0001);// good-> good;
  dbMask.push_back(0x07FE);// status 1..10 -> problematic
  dbMask.push_back(0x0000);// nothing->recovered
  dbMask.push_back(0x0000);// nothing->time
  dbMask.push_back(0x0000);// nothing->weird
  dbMask.push_back(0xFC00);// status 11..16 ->bad


  ps.addParameter< std::vector<uint32_t> > ("flagMask",fMask);
  ps.addParameter< std::vector<uint32_t> > ("dbstatusMask",dbMask);
  
  
  algo_=new EcalSeverityLevelAlgo(ps);
}
void testEcalSeverityLevelAlgo::testSeverity(){

  EBDetId  id(1,1);
  EcalRecHit rh1(id,0,0);
  rh1.setFlag(EcalRecHit::kGood);
   
  CPPUNIT_ASSERT(algo_->severityLevel(rh1) == EcalSeverityLevelAlgo::kGood);
  
  EcalRecHit rh2(id,0,0);
  rh2.setFlag(EcalRecHit::kPoorReco);
  rh2.setFlag(EcalRecHit::kPoorCalib);
 
  CPPUNIT_ASSERT(algo_->severityLevel(rh2) == EcalSeverityLevelAlgo::kProblematic);

  EcalRecHit rh3(id,0,0);
  rh3.setFlag(EcalRecHit::kLeadingEdgeRecovered);
  rh3.setFlag(EcalRecHit::kTowerRecovered);
 
  CPPUNIT_ASSERT(algo_->severityLevel(rh3) == EcalSeverityLevelAlgo::kRecovered);


  EcalRecHit rh4(id,0,0);
  rh4.setFlag(EcalRecHit::kOutOfTime);
  rh4.setFlag(EcalRecHit::kTowerRecovered);
  
  CPPUNIT_ASSERT(algo_->severityLevel(rh4) == EcalSeverityLevelAlgo::kTime);

  EcalRecHit rh5(id,0,0);
  rh5.setFlag(EcalRecHit::kWeird);
  rh5.setFlag(EcalRecHit::kDiWeird);
  
  CPPUNIT_ASSERT(algo_->severityLevel(rh5) == EcalSeverityLevelAlgo::kWeird);
  
  EcalRecHit rh6(id,0,0);
  rh6.setFlag(EcalRecHit::kFaultyHardware);
  rh6.setFlag(EcalRecHit::kNoisy);
  rh6.setFlag(EcalRecHit::kSaturated);
  rh6.setFlag(EcalRecHit::kDead);
  rh6.setFlag(EcalRecHit::kKilled);

  CPPUNIT_ASSERT(algo_->severityLevel(rh6) == EcalSeverityLevelAlgo::kBad);
  
}
