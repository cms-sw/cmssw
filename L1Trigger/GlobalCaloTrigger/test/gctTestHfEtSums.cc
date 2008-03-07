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
  m_expectedRing0EtSumPositiveEta = 0;
  m_expectedRing0EtSumNegativeEta = 0;
  m_expectedRing1EtSumPositiveEta = 0;
  m_expectedRing1EtSumNegativeEta = 0;
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
  static const unsigned NUMBER_OF_INNER_RINGS=2;
  static const unsigned NUMBER_OF_RINGS_PER_WHEEL=L1CaloRegionDetId::N_ETA/2;
  static const unsigned MIN_ETA_COUNTS =NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_FRWRD_RINGS;
  static const unsigned MIN_ETA_HF_SUMS=NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_INNER_RINGS;
  static const unsigned MAX_ETSUM_VALUE = L1GctJetCounts::kEtHfSumMaxValue;
  static const unsigned MAX_TOWER_COUNT = 31;

  std::vector<unsigned> etNegativeEta(NUMBER_OF_INNER_RINGS,0);
  std::vector<unsigned> etPositiveEta(NUMBER_OF_INNER_RINGS,0);
  std::vector<bool>     ofNegativeEta(NUMBER_OF_INNER_RINGS,false);
  std::vector<bool>     ofPositiveEta(NUMBER_OF_INNER_RINGS,false);
  // Loop over regions, selecting those in the outer ring(s) of the Hf
  for (RegionsVector::const_iterator region=inputRegions.begin(); region!=inputRegions.end(); region++) {
    if (region->id().rctEta() >= MIN_ETA_HF_SUMS) {
      unsigned ring = NUMBER_OF_RINGS_PER_WHEEL - region->id().rctEta() - 1;
      // Split into positive and negative eta
      if (region->id().ieta() < NUMBER_OF_RINGS_PER_WHEEL) {
        etNegativeEta.at(ring) += region->et();
        ofNegativeEta.at(ring) = ofNegativeEta.at(ring) || region->overFlow();
      } else {
        etPositiveEta.at(ring) += region->et();
        ofPositiveEta.at(ring) = ofPositiveEta.at(ring) || region->overFlow();
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
  m_expectedRing0EtSumNegativeEta = ( (etNegativeEta.at(0) > MAX_ETSUM_VALUE) || ofNegativeEta.at(0) ? MAX_ETSUM_VALUE
                                                                                                     : etNegativeEta.at(0) );
  m_expectedRing0EtSumPositiveEta = ( (etPositiveEta.at(0) > MAX_ETSUM_VALUE) || ofPositiveEta.at(0) ? MAX_ETSUM_VALUE
                                                                                                     : etPositiveEta.at(0) );
  m_expectedRing1EtSumNegativeEta = ( (etNegativeEta.at(1) > MAX_ETSUM_VALUE) || ofNegativeEta.at(1) ? MAX_ETSUM_VALUE
                                                                                                     : etNegativeEta.at(1) );
  m_expectedRing1EtSumPositiveEta = ( (etPositiveEta.at(1) > MAX_ETSUM_VALUE) || ofPositiveEta.at(1) ? MAX_ETSUM_VALUE
                                                                                                     : etPositiveEta.at(1) );
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

  unsigned towerCountPositiveEta = jetCounts.at(6);
  unsigned towerCountNegativeEta = jetCounts.at(7);
  unsigned etSumRing0PositiveEta = jetCounts.at(8);
  unsigned etSumRing0NegativeEta = jetCounts.at(9);
  unsigned etSumRing1PositiveEta = jetCounts.at(10);
  unsigned etSumRing1NegativeEta = jetCounts.at(11);

  if (etSumRing0PositiveEta != m_expectedRing0EtSumPositiveEta)
    { cout << "Hf Et Sum Positive Eta, expected " << m_expectedRing0EtSumPositiveEta 
                                    << ", found " << etSumRing0PositiveEta << endl;
      testPass = false; }
  if (etSumRing0NegativeEta != m_expectedRing0EtSumNegativeEta )
    { cout << "Hf Et Sum Negative Eta, expected " << m_expectedRing0EtSumNegativeEta 
                                    << ", found " << etSumRing0NegativeEta << endl;
      testPass = false; }
  if (etSumRing1PositiveEta != m_expectedRing1EtSumPositiveEta)
    { cout << "Hf Et Sum Positive Eta, expected " << m_expectedRing1EtSumPositiveEta 
                                    << ", found " << etSumRing1PositiveEta << endl;
      testPass = false; }
  if (etSumRing1NegativeEta != m_expectedRing1EtSumNegativeEta )
    { cout << "Hf Et Sum Negative Eta, expected " << m_expectedRing1EtSumNegativeEta 
                                    << ", found " << etSumRing1NegativeEta << endl;
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

