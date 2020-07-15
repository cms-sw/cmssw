#include "L1Trigger/GlobalCaloTrigger/test/gctTestHfEtSums.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalHfSumAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestHfEtSums::gctTestHfEtSums()
    : m_etScale(),
      m_expectedRing0EtSumPositiveEta(),
      m_expectedRing0EtSumNegativeEta(),
      m_expectedRing1EtSumPositiveEta(),
      m_expectedRing1EtSumNegativeEta(),
      m_expectedRing0BitCountPositiveEta(),
      m_expectedRing0BitCountNegativeEta(),
      m_expectedRing1BitCountPositiveEta(),
      m_expectedRing1BitCountNegativeEta() {}

gctTestHfEtSums::~gctTestHfEtSums() {}

//=================================================================================================================
//
/// Configuration

void gctTestHfEtSums::configure(const L1CaloEtScale* scale) { m_etScale = scale; }

bool gctTestHfEtSums::setupOk() const { return (m_etScale != nullptr); }
//=================================================================================================================
//
/// Reset stored sums
void gctTestHfEtSums::reset() {
  m_expectedRing0EtSumPositiveEta.clear();
  m_expectedRing0EtSumNegativeEta.clear();
  m_expectedRing1EtSumPositiveEta.clear();
  m_expectedRing1EtSumNegativeEta.clear();
  m_expectedRing0BitCountPositiveEta.clear();
  m_expectedRing0BitCountNegativeEta.clear();
  m_expectedRing1BitCountPositiveEta.clear();
  m_expectedRing1BitCountNegativeEta.clear();
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
void gctTestHfEtSums::fillExpectedHfSums(const std::vector<RegionsVector>& inputRegions) {
  // A bunch of constants defining how the code works
  // Some now unused - comment to avoid compiler warnings
  //static const unsigned NUMBER_OF_FRWRD_RINGS=4;
  static const unsigned NUMBER_OF_INNER_RINGS = 2;
  static const unsigned NUMBER_OF_RINGS_PER_WHEEL = L1CaloRegionDetId::N_ETA / 2;
  //static const unsigned MIN_ETA_COUNTS =NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_FRWRD_RINGS;
  static const unsigned MIN_ETA_HF_SUMS = NUMBER_OF_RINGS_PER_WHEEL - NUMBER_OF_INNER_RINGS;
  // TODO - put these bit sizes somewhere
  static const unsigned MAX_ETSUM_VALUE = (1 << 8) - 1;
  static const unsigned MAX_TOWER_COUNT = (1 << 5) - 1;

  unsigned numOfBx = inputRegions.size();
  m_expectedRing0EtSumPositiveEta.resize(numOfBx);
  m_expectedRing0EtSumNegativeEta.resize(numOfBx);
  m_expectedRing1EtSumPositiveEta.resize(numOfBx);
  m_expectedRing1EtSumNegativeEta.resize(numOfBx);
  m_expectedRing0BitCountPositiveEta.resize(numOfBx);
  m_expectedRing0BitCountNegativeEta.resize(numOfBx);
  m_expectedRing1BitCountPositiveEta.resize(numOfBx);
  m_expectedRing1BitCountNegativeEta.resize(numOfBx);

  for (unsigned bx = 0; bx < inputRegions.size(); bx++) {
    std::vector<unsigned> etNegativeEta(NUMBER_OF_INNER_RINGS, 0);
    std::vector<unsigned> etPositiveEta(NUMBER_OF_INNER_RINGS, 0);
    std::vector<bool> ofNegativeEta(NUMBER_OF_INNER_RINGS, false);
    std::vector<bool> ofPositiveEta(NUMBER_OF_INNER_RINGS, false);
    std::vector<unsigned> bcNegativeEta(NUMBER_OF_INNER_RINGS, 0);
    std::vector<unsigned> bcPositiveEta(NUMBER_OF_INNER_RINGS, 0);
    // Loop over regions, selecting those in the outer ring(s) of the Hf
    for (RegionsVector::const_iterator region = inputRegions.at(bx).begin(); region != inputRegions.at(bx).end();
         region++) {
      if (region->id().rctEta() >= MIN_ETA_HF_SUMS) {
        unsigned ring = NUMBER_OF_RINGS_PER_WHEEL - region->id().rctEta() - 1;
        // Split into positive and negative eta
        if (region->id().ieta() < NUMBER_OF_RINGS_PER_WHEEL) {
          etNegativeEta.at(ring) += region->et();
          ofNegativeEta.at(ring) = ofNegativeEta.at(ring) || region->overFlow();
          if (region->fineGrain())
            ++bcNegativeEta.at(ring);
        } else {
          etPositiveEta.at(ring) += region->et();
          ofPositiveEta.at(ring) = ofPositiveEta.at(ring) || region->overFlow();
          if (region->fineGrain())
            ++bcPositiveEta.at(ring);
        }
      }
    }
    m_expectedRing0EtSumNegativeEta.at(bx) =
        ((etNegativeEta.at(0) > MAX_ETSUM_VALUE) || ofNegativeEta.at(0) ? MAX_ETSUM_VALUE : etNegativeEta.at(0));
    m_expectedRing0EtSumPositiveEta.at(bx) =
        ((etPositiveEta.at(0) > MAX_ETSUM_VALUE) || ofPositiveEta.at(0) ? MAX_ETSUM_VALUE : etPositiveEta.at(0));
    m_expectedRing1EtSumNegativeEta.at(bx) =
        ((etNegativeEta.at(1) > MAX_ETSUM_VALUE) || ofNegativeEta.at(1) ? MAX_ETSUM_VALUE : etNegativeEta.at(1));
    m_expectedRing1EtSumPositiveEta.at(bx) =
        ((etPositiveEta.at(1) > MAX_ETSUM_VALUE) || ofPositiveEta.at(1) ? MAX_ETSUM_VALUE : etPositiveEta.at(1));
    m_expectedRing0BitCountNegativeEta.at(bx) = bcNegativeEta.at(0);
    m_expectedRing0BitCountPositiveEta.at(bx) = bcPositiveEta.at(0);
    m_expectedRing1BitCountNegativeEta.at(bx) = bcNegativeEta.at(1);
    m_expectedRing1BitCountPositiveEta.at(bx) = bcPositiveEta.at(1);
    if (m_expectedRing0BitCountNegativeEta.at(bx) > MAX_TOWER_COUNT)
      m_expectedRing0BitCountNegativeEta.at(bx) = MAX_TOWER_COUNT;
    if (m_expectedRing0BitCountPositiveEta.at(bx) > MAX_TOWER_COUNT)
      m_expectedRing0BitCountPositiveEta.at(bx) = MAX_TOWER_COUNT;
    if (m_expectedRing1BitCountNegativeEta.at(bx) > MAX_TOWER_COUNT)
      m_expectedRing1BitCountNegativeEta.at(bx) = MAX_TOWER_COUNT;
    if (m_expectedRing1BitCountPositiveEta.at(bx) > MAX_TOWER_COUNT)
      m_expectedRing1BitCountPositiveEta.at(bx) = MAX_TOWER_COUNT;
  }
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestHfEtSums::checkHfEtSums(const L1GlobalCaloTrigger* gct, const int numOfBx) const {
  if (!setupOk()) {
    cout << "checkHfEtSums setup check failed" << endl;
    return false;
  }

  bool testPass = true;

  for (int bx = 0; bx < numOfBx; bx++) {
    unsigned bitCountRing0PositiveEta =
        gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::bitCountPosEtaRing1).at(bx);
    unsigned bitCountRing0NegativeEta =
        gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::bitCountNegEtaRing1).at(bx);
    unsigned bitCountRing1PositiveEta =
        gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::bitCountPosEtaRing2).at(bx);
    unsigned bitCountRing1NegativeEta =
        gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::bitCountNegEtaRing2).at(bx);
    unsigned etSumRing0PositiveEta = gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::etSumPosEtaRing1).at(bx);
    unsigned etSumRing0NegativeEta = gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::etSumNegEtaRing1).at(bx);
    unsigned etSumRing1PositiveEta = gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::etSumPosEtaRing2).at(bx);
    unsigned etSumRing1NegativeEta = gct->getHfSumProcessor()->hfSumsOutput(L1GctHfEtSumsLut::etSumNegEtaRing2).at(bx);

    if (etSumRing0PositiveEta != etSumLut(m_expectedRing0EtSumPositiveEta.at(bx))) {
      cout << "Hf Et Sum Positive Eta, expected " << etSumLut(m_expectedRing0EtSumPositiveEta.at(bx)) << ", found "
           << etSumRing0PositiveEta << endl;
      testPass = false;
    }
    if (etSumRing0NegativeEta != etSumLut(m_expectedRing0EtSumNegativeEta.at(bx))) {
      cout << "Hf Et Sum Negative Eta, expected " << etSumLut(m_expectedRing0EtSumNegativeEta.at(bx)) << ", found "
           << etSumRing0NegativeEta << endl;
      testPass = false;
    }
    if (etSumRing1PositiveEta != etSumLut(m_expectedRing1EtSumPositiveEta.at(bx))) {
      cout << "Hf Et Sum Positive Eta, expected " << etSumLut(m_expectedRing1EtSumPositiveEta.at(bx)) << ", found "
           << etSumRing1PositiveEta << endl;
      testPass = false;
    }
    if (etSumRing1NegativeEta != etSumLut(m_expectedRing1EtSumNegativeEta.at(bx))) {
      cout << "Hf Et Sum Negative Eta, expected " << etSumLut(m_expectedRing1EtSumNegativeEta.at(bx)) << ", found "
           << etSumRing1NegativeEta << endl;
      testPass = false;
    }
    if (bitCountRing0PositiveEta != countLut(m_expectedRing0BitCountPositiveEta.at(bx))) {
      cout << "000Hf Tower Count Positive Eta, expected " << countLut(m_expectedRing0BitCountPositiveEta.at(bx))
           << ", found " << bitCountRing0PositiveEta << endl;
      testPass = false;
    }
    if (bitCountRing0NegativeEta != countLut(m_expectedRing0BitCountNegativeEta.at(bx))) {
      cout << "111Hf Tower Count Negative Eta, expected " << countLut(m_expectedRing0BitCountNegativeEta.at(bx))
           << ", found " << bitCountRing0NegativeEta << endl;
      testPass = false;
    }
    if (bitCountRing1PositiveEta != countLut(m_expectedRing1BitCountPositiveEta.at(bx))) {
      cout << "222Hf Tower Count Positive Eta, expected " << countLut(m_expectedRing1BitCountPositiveEta.at(bx))
           << ", found " << bitCountRing1PositiveEta << endl;
      testPass = false;
    }
    if (bitCountRing1NegativeEta != countLut(m_expectedRing1BitCountNegativeEta.at(bx))) {
      cout << "333Hf Tower Count Negative Eta, expected " << countLut(m_expectedRing1BitCountNegativeEta.at(bx))
           << ", found " << bitCountRing1NegativeEta << endl;
      testPass = false;
    }

    // end of loop over bunch crossings
  }
  return testPass;
}

unsigned gctTestHfEtSums::etSumLut(const unsigned expectedValue) const {
  // Get the rank from the relevant scale object
  return m_etScale->rank((uint16_t)expectedValue);
}

unsigned gctTestHfEtSums::countLut(const unsigned expectedValue) const {
  // Bit count scale is hard coded with 1<->1 match between input and output
  // TODO - put these bit sizes somewhere
  static const unsigned maxLut = (1 << 3) - 1;
  unsigned result = maxLut;
  unsigned bitCount = expectedValue;
  if (bitCount < maxLut)
    result = bitCount;
  return result;
}
