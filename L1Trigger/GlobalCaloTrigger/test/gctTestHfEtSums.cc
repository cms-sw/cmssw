#include "L1Trigger/GlobalCaloTrigger/test/gctTestHfEtSums.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <math.h>
#include <iostream>
#include <cassert>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestHfEtSums::gctTestHfEtSums() {}
gctTestHfEtSums::~gctTestHfEtSums() {}

//=================================================================================================================
//
  /// Reset stored sums
void gctTestHfEtSums::reset() {
  m_expectedEtSumPositiveEta      = 0;
  m_expectedEtSumNegativeEta      = 0;
  m_expectedTowerCountPositiveEta = 0;
  m_expectedTowerCountNegativeEta = 0;
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
void gctTestHfEtSums::fillExpectedHfSums(const RegionsVector& inputRegions)
{
  static const unsigned NUMBER_OF_INNER_RINGS=1;
  static const unsigned NUMBER_OF_RINGS_PER_WHEEL=L1CaloRegionDetId::N_ETA/2;
  static const unsigned MIN_ETA_HF_SUMS=NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_INNER_RINGS;
  static const unsigned BIT_SHIFT = L1GctJetCounts::kEtHfSumBitShift;

  // Loop over regions, selecting those in the outer ring(s) of the Hf
  for (RegionsVector::const_iterator region=inputRegions.begin(); region!=inputRegions.end(); region++) {
    if (region->id().rctEta() >= MIN_ETA_HF_SUMS) {
      // Split into positive and negative eta
      if (region->id().ieta() < NUMBER_OF_RINGS_PER_WHEEL) {
        m_expectedEtSumNegativeEta += region->et() >> BIT_SHIFT;
        if (region->fineGrain()) ++m_expectedTowerCountNegativeEta;
      } else {
        m_expectedEtSumPositiveEta += region->et() >> BIT_SHIFT;
        if (region->fineGrain()) ++m_expectedTowerCountPositiveEta;
      }
    }
  }
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestHfEtSums::checkHfEtSums(const L1GlobalCaloTrigger* gct) const
{
  bool testPass = true;

  // Get the jet count bits
  std::vector< unsigned > jetCounts=gct->getEnergyFinalStage()->getJetCountValues();
  assert (jetCounts.size()==12);

  unsigned jetCountWord1= jetCounts.at(6)         | (jetCounts.at(7)  <<  5) | (jetCounts.at(8)  << 10) |
                         (jetCounts.at(9)  << 16) | (jetCounts.at(10) << 21) | (jetCounts.at(11) << 26) ;

  unsigned towerCountPositiveEta = jetCounts.at(6);
  unsigned towerCountNegativeEta = jetCounts.at(7);
  unsigned etSumPositiveEta = (jetCountWord1 >> 16) & L1GctJetCounts::kEtHfSumMaxValue;
  unsigned etSumNegativeEta = (jetCountWord1 >> 24) & L1GctJetCounts::kEtHfSumMaxValue;

  if (etSumPositiveEta != m_expectedEtSumPositiveEta)
    { cout << "Et Sum Positive Eta, expected " << m_expectedEtSumPositiveEta
                                 << ", found " << etSumPositiveEta << endl;
      testPass = false; }
  if (etSumNegativeEta != m_expectedEtSumNegativeEta)
    { cout << "Et Sum Negative Eta, expected " << m_expectedEtSumNegativeEta
                                 << ", found " << etSumNegativeEta << endl;
      testPass = false; }
  if (towerCountPositiveEta != m_expectedTowerCountPositiveEta)
    { cout << "Tower Count Positive Eta, expected " << m_expectedTowerCountPositiveEta
                                      << ", found " << towerCountPositiveEta << endl;
      testPass = false; }
  if (towerCountNegativeEta != m_expectedTowerCountNegativeEta)
    { cout << "Tower Count Negative Eta, expected " << m_expectedTowerCountNegativeEta
                                      << ", found " << towerCountNegativeEta << endl;
      testPass = false; }

  return testPass;
}

