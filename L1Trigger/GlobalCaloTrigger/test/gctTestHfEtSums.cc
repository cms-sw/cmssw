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
  // A bunch of constants defining how the code works
  static const unsigned NUMBER_OF_FRWRD_RINGS=4;
  static const unsigned NUMBER_OF_INNER_RINGS=1;
  static const unsigned NUMBER_OF_RINGS_PER_WHEEL=L1CaloRegionDetId::N_ETA/2;
  static const unsigned MIN_ETA_COUNTS =NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_FRWRD_RINGS;
  static const unsigned MIN_ETA_HF_SUMS=NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_INNER_RINGS;
  static const unsigned BIT_SHIFT = L1GctJetCounts::kEtHfSumBitShift;
  static const unsigned MAX_ETSUM_VALUE = L1GctJetCounts::kEtHfSumMaxValue;
  static const unsigned MAX_TOWER_COUNT = 31;

  bool overFlowNegativeEta = false;
  bool overFlowPositiveEta = false;
  // Loop over regions, selecting those in the outer ring(s) of the Hf
  for (RegionsVector::const_iterator region=inputRegions.begin(); region!=inputRegions.end(); region++) {
    if (region->id().rctEta() >= MIN_ETA_HF_SUMS) {
      // Split into positive and negative eta
      if (region->id().ieta() < NUMBER_OF_RINGS_PER_WHEEL) {
        m_expectedEtSumNegativeEta += region->et() >> BIT_SHIFT;
        overFlowNegativeEta |= region->overFlow();
      } else {
        m_expectedEtSumPositiveEta += region->et() >> BIT_SHIFT;
        overFlowPositiveEta |= region->overFlow();
      }
    }
    if (region->id().rctEta() >= MIN_ETA_COUNTS) {
      // Split into positive and negative eta
      if (region->id().ieta() < NUMBER_OF_RINGS_PER_WHEEL) {
        if (region->fineGrain()) ++m_expectedTowerCountNegativeEta;
      } else {
        if (region->fineGrain()) ++m_expectedTowerCountPositiveEta;
      }
    }
  }
  if (m_expectedEtSumNegativeEta > MAX_ETSUM_VALUE || overFlowNegativeEta) m_expectedEtSumNegativeEta = MAX_ETSUM_VALUE;
  if (m_expectedEtSumPositiveEta > MAX_ETSUM_VALUE || overFlowPositiveEta) m_expectedEtSumPositiveEta = MAX_ETSUM_VALUE;
  if (m_expectedTowerCountNegativeEta > MAX_TOWER_COUNT) m_expectedTowerCountNegativeEta = MAX_TOWER_COUNT;
  if (m_expectedTowerCountPositiveEta > MAX_TOWER_COUNT) m_expectedTowerCountPositiveEta = MAX_TOWER_COUNT;
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
  unsigned etSumPositiveEta = (((jetCountWord1 & 0x03e00000) >> 19) | ((jetCountWord1 & 0x00030000) >> 16));
  unsigned etSumNegativeEta = (((jetCountWord1 & 0x7c000000) >> 24) | ((jetCountWord1 & 0x000c0000) >> 18));

  if (etSumPositiveEta != m_expectedEtSumPositiveEta)
    { cout << "Hf Et Sum Positive Eta, expected " << m_expectedEtSumPositiveEta
                                    << ", found " << etSumPositiveEta << endl;
      testPass = false; }
  if (etSumNegativeEta != m_expectedEtSumNegativeEta)
    { cout << "Hf Et Sum Negative Eta, expected " << m_expectedEtSumNegativeEta
                                    << ", found " << etSumNegativeEta << endl;
      testPass = false; }
  if (towerCountPositiveEta != m_expectedTowerCountPositiveEta)
    { cout << "Hf Tower Count Positive Eta, expected " << m_expectedTowerCountPositiveEta
                                         << ", found " << towerCountPositiveEta << endl;
      testPass = false; }
  if (towerCountNegativeEta != m_expectedTowerCountNegativeEta)
    { cout << "Hf Tower Count Negative Eta, expected " << m_expectedTowerCountNegativeEta
                                         << ", found " << towerCountNegativeEta << endl;
      testPass = false; }

  return testPass;
}

