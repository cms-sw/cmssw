/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class testEcalSeverityLevelAlgo : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEcalSeverityLevelAlgo);
  CPPUNIT_TEST(testSeverity);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override { delete algo_; }

  void testSeverity();

  EcalSeverityLevelAlgo* algo_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalSeverityLevelAlgo);

using std::string;

void testEcalSeverityLevelAlgo::setUp() {
  edm::ParameterSet ps;

  ps.addParameter<double>("timeThresh", 2.0);

  edm::ParameterSet flagmaskps;
  std::vector<std::string> kGoodv, kProblematicv, kRecoveredv, kTimev, kWeirdv, kBadv;
  kGoodv.push_back("kGood");
  kProblematicv.push_back("kPoorReco");
  kProblematicv.push_back("kPoorCalib");
  kProblematicv.push_back("kNoisy");
  kProblematicv.push_back("kSaturated");
  kRecoveredv.push_back("kLeadingEdgeRecovered");
  kRecoveredv.push_back("kTowerRecovered");
  kTimev.push_back("kOutOfTime");
  kWeirdv.push_back("kWeird");
  kWeirdv.push_back("kDiWeird");
  kBadv.push_back("kFaultyHardware");
  kBadv.push_back("kDead");
  kBadv.push_back("kKilled");

  flagmaskps.addParameter<std::vector<string> >("kGood", kGoodv);
  flagmaskps.addParameter<std::vector<string> >("kProblematic", kProblematicv);
  flagmaskps.addParameter<std::vector<string> >("kRecovered", kRecoveredv);
  flagmaskps.addParameter<std::vector<string> >("kTime", kTimev);
  flagmaskps.addParameter<std::vector<string> >("kWeird", kWeirdv);
  flagmaskps.addParameter<std::vector<string> >("kBad", kBadv);

  ps.addParameter<edm::ParameterSet>("flagMask", flagmaskps);

  edm::ParameterSet dbmaskps;
  std::vector<uint32_t> kGoods, kProblematics, kRecovereds, kTimes, kWeirds, kBads;

  kGoods.push_back(0);
  for (int i = 1; i <= 10; ++i)
    kProblematics.push_back(i);
  for (int i = 11; i <= 16; ++i)
    kBads.push_back(i);

  ps.addParameter<edm::ParameterSet>("dbstatusMask", dbmaskps);

  algo_ = new EcalSeverityLevelAlgo(ps);
}
void testEcalSeverityLevelAlgo::testSeverity() {
  EBDetId id(1, 1);
  EcalRecHit rh1(id, 0, 0);
  rh1.setFlag(EcalRecHit::kGood);

  CPPUNIT_ASSERT(algo_->severityLevel(rh1) == EcalSeverityLevel::kGood);

  EcalRecHit rh2(id, 0, 0);
  rh2.setFlag(EcalRecHit::kPoorReco);
  rh2.setFlag(EcalRecHit::kPoorCalib);
  rh2.setFlag(EcalRecHit::kNoisy);
  rh2.setFlag(EcalRecHit::kSaturated);
  CPPUNIT_ASSERT(algo_->severityLevel(rh2) == EcalSeverityLevel::kProblematic);
  CPPUNIT_ASSERT(algo_->severityLevel(rh2) != EcalSeverityLevel::kBad);
  CPPUNIT_ASSERT(algo_->severityLevel(rh2) != EcalSeverityLevel::kGood);
  CPPUNIT_ASSERT(algo_->severityLevel(rh2) != EcalSeverityLevel::kWeird);

  EcalRecHit rh3(id, 0, 0);
  rh3.setFlag(EcalRecHit::kLeadingEdgeRecovered);
  rh3.setFlag(EcalRecHit::kTowerRecovered);

  CPPUNIT_ASSERT(algo_->severityLevel(rh3) == EcalSeverityLevel::kRecovered);

  EcalRecHit rh4(id, 5.0, 0);
  rh4.setFlag(EcalRecHit::kOutOfTime);
  rh4.setFlag(EcalRecHit::kTowerRecovered);

  CPPUNIT_ASSERT(algo_->severityLevel(rh4) == EcalSeverityLevel::kTime);

  EcalRecHit rh5(id, 0, 0);
  rh5.setFlag(EcalRecHit::kWeird);
  rh5.setFlag(EcalRecHit::kDiWeird);

  CPPUNIT_ASSERT(algo_->severityLevel(rh5) == EcalSeverityLevel::kWeird);

  EcalRecHit rh6(id, 0, 0);
  rh6.setFlag(EcalRecHit::kFaultyHardware);
  rh6.setFlag(EcalRecHit::kDead);
  rh6.setFlag(EcalRecHit::kKilled);

  CPPUNIT_ASSERT(algo_->severityLevel(rh6) == EcalSeverityLevel::kBad);

  EcalRecHit rh7(id, 1.5, 0);
  rh7.setFlag(EcalRecHit::kOutOfTime);

  CPPUNIT_ASSERT(algo_->severityLevel(rh7) == EcalSeverityLevel::kGood);

  EcalRecHit rh8(id, 2.5, 0);
  rh8.setFlag(EcalRecHit::kOutOfTime);

  CPPUNIT_ASSERT(algo_->severityLevel(rh8) == EcalSeverityLevel::kTime);
}
