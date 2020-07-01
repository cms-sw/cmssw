#include "L1Trigger/GlobalCaloTrigger/test/gctTestHt.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestHt::gctTestHt()
    : m_bxStart(), m_numOfBx(1), minusWheelJetDta(), plusWheelJetData(), m_jetEtScale(), m_htMissScale(), m_jfPars() {}

gctTestHt::~gctTestHt() {}

//=================================================================================================================
//
/// Configuration

void gctTestHt::configure(const L1CaloEtScale* jetScale,
                          const L1CaloEtScale* mhtScale,
                          const L1GctJetFinderParams* jfPars) {
  m_jetEtScale = jetScale;
  m_htMissScale = mhtScale;
  m_jfPars = jfPars;
}

bool gctTestHt::setupOk() const { return (m_jetEtScale != nullptr && m_htMissScale != nullptr && m_jfPars != nullptr); }

//=================================================================================================================
//
/// Read the input jet data from the jetfinders (after GCT processing).
void gctTestHt::fillRawJetData(const L1GlobalCaloTrigger* gct) {
  minusWheelJetDta.clear();
  plusWheelJetData.clear();
  minusWheelJetDta.resize(9 * m_numOfBx);
  plusWheelJetData.resize(9 * m_numOfBx);

  int bx = m_bxStart;
  unsigned mPos = 0;
  unsigned pPos = 0;
  for (int ibx = 0; ibx < m_numOfBx; ibx++) {
    int leaf = 0;

    // Minus Wheel
    for (; leaf < 3; leaf++) {
      minusWheelJetDta.at(mPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA(), mPos % 9, bx);
      mPos++;
      minusWheelJetDta.at(mPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB(), mPos % 9, bx);
      mPos++;
      minusWheelJetDta.at(mPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC(), mPos % 9, bx);
      mPos++;
    }

    // Plus Wheel
    for (; leaf < 6; leaf++) {
      plusWheelJetData.at(pPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA(), pPos % 9, bx);
      pPos++;
      plusWheelJetData.at(pPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB(), pPos % 9, bx);
      pPos++;
      plusWheelJetData.at(pPos) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC(), pPos % 9, bx);
      pPos++;
    }

    bx++;
  }
}

//=================================================================================================================
//
/// Set array sizes for the number of bunch crossings
void gctTestHt::setBxRange(const int bxStart, const int numOfBx) {
  // Allow the start of the bunch crossing range to be altered
  // without losing previously stored jet data
  rawJetData temp;
  for (int bx = bxStart; bx < m_bxStart; bx++) {
    minusWheelJetDta.insert(minusWheelJetDta.begin(), 9, temp);
    plusWheelJetData.insert(plusWheelJetData.begin(), 9, temp);
  }

  m_bxStart = bxStart;

  // Resize the vectors without clearing previously stored values
  minusWheelJetDta.resize(9 * numOfBx);
  plusWheelJetData.resize(9 * numOfBx);
  m_numOfBx = numOfBx;
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestHt::checkHtSums(const L1GlobalCaloTrigger* gct) const {
  if (!setupOk()) {
    cout << "checkHtSums setup check failed" << endl;
    return false;
  }

  bool testPass = true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  L1GctInternJetDataCollection internalJets = gct->getInternalJets();

  std::vector<rawJetData>::const_iterator mJet = minusWheelJetDta.begin();
  std::vector<rawJetData>::const_iterator pJet = plusWheelJetData.begin();

  for (int bx = 0; bx < m_numOfBx; bx++) {
    unsigned htMinusVl = 0;
    unsigned htPlusVal = 0;
    int hxMinusVl = 0;
    int hxPlusVal = 0;
    int hyMinusVl = 0;
    int hyPlusVal = 0;
    bool httMinusInputOf = false;
    bool httPlusInputOvf = false;
    bool htmMinusInputOf = false;
    bool htmPlusInputOvf = false;

    //
    // Check the Ht calculation (starting from the found jets)
    //--------------------------------------------------------------------------------------
    //
    // Minus Wheel
    int leaf = 0;
    for (; leaf < 3; leaf++) {
      unsigned leafHttSum = 0;
      int leafHtxSum = 0;
      int leafHtySum = 0;
      bool leafHttOvf = false;
      bool leafHtmOvf = false;

      for (int jf = 0; jf < 3; jf++) {
        assert(mJet != minusWheelJetDta.end());
        leafHttSum += (mJet->httSum);
        leafHttOvf |= (mJet->httOverFlow);
        leafHtxSum += (mJet->htxSum);
        leafHtySum += (mJet->htySum);
        leafHtmOvf |= (mJet->htmOverFlow);
        mJet++;
      }
      if (leafHttSum >= 4096 || leafHttOvf) {
        leafHttSum = 4095;
        leafHttOvf = true;
      }
      if (leafHttSum == gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).value()) {
        htMinusVl += leafHttSum;
      } else {
        cout << "Ht sum check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
      if (leafHtxSum == gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).value()) {
        hxMinusVl += leafHtxSum;
      } else {
        cout << "Hx sum check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
      if (leafHtySum == gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).value()) {
        hyMinusVl += leafHtySum;
      } else {
        cout << "Hy sum check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
      if ((gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).overFlow() == leafHttOvf) &&
          (gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).overFlow() == leafHtmOvf) &&
          (gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).overFlow() == leafHtmOvf)) {
        httMinusInputOf |= leafHttOvf;
        htmMinusInputOf |= leafHtmOvf;
      } else {
        cout << "Ht minus overflow check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
    }

    // Plus Wheel
    for (; leaf < 6; leaf++) {
      unsigned leafHttSum = 0;
      int leafHtxSum = 0;
      int leafHtySum = 0;
      bool leafHttOvf = false;
      bool leafHtmOvf = false;
      for (int jf = 0; jf < 3; jf++) {
        assert(pJet != plusWheelJetData.end());
        leafHttSum += (pJet->httSum);
        leafHttOvf |= (pJet->httOverFlow);
        leafHtxSum += (pJet->htxSum);
        leafHtySum += (pJet->htySum);
        leafHtmOvf |= (pJet->htmOverFlow);
        pJet++;
      }
      if (leafHttSum >= 4096 || leafHttOvf) {
        leafHttSum = 4095;
        leafHttOvf = true;
      }
      if (leafHttSum == gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).value()) {
        htPlusVal += leafHttSum;
      } else {
        cout << "Ht sum check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
      if (leafHtxSum == gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).value()) {
        hxPlusVal += leafHtxSum;
      } else {
        cout << "Hx sum check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
      if (leafHtySum == gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).value()) {
        hyPlusVal += leafHtySum;
      } else {
        cout << "Hy sum check leaf " << leaf << endl;
        testPass = false;
      }
      if ((gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).overFlow() == leafHttOvf) &&
          (gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).overFlow() == leafHtmOvf) &&
          (gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).overFlow() == leafHtmOvf)) {
        httPlusInputOvf |= leafHttOvf;
        htmPlusInputOvf |= leafHtmOvf;
      } else {
        cout << "Ht plus overflow check leaf " << leaf << " bx " << bx << endl;
        testPass = false;
      }
    }

    unsigned htTotal = htMinusVl + htPlusVal;

    bool httMinusOvrFlow = (htMinusVl >= 4096) || httMinusInputOf;
    bool httPlusOverFlow = (htPlusVal >= 4096) || httPlusInputOvf;

    if (httMinusOvrFlow)
      htMinusVl = 4095;
    if (httPlusOverFlow)
      htPlusVal = 4095;

    bool httTotalOvrFlow = (htTotal >= 4096) || httMinusOvrFlow || httPlusOverFlow;

    if (httTotalOvrFlow)
      htTotal = 4095;

    int hxTotal = hxMinusVl + hxPlusVal;
    int hyTotal = hyMinusVl + hyPlusVal;

    bool htmMinusOvrFlow = htmMinusInputOf;
    bool htmPlusOverFlow = htmPlusInputOvf;
    // Check the input to the final GlobalEnergyAlgos is as expected
    //--------------------------------------------------------------------------------------
    //
    if (!myGlobalEnergy->getInputHtVlMinusWheel().at(bx).overFlow() && !httMinusOvrFlow &&
        (myGlobalEnergy->getInputHtVlMinusWheel().at(bx).value() != htMinusVl)) {
      cout << "ht Minus " << htMinusVl << " bx " << bx << endl;
      testPass = false;
    }

    if (myGlobalEnergy->getInputHtVlMinusWheel().at(bx).value() != htMinusVl) {
      cout << "ht Minus ovF " << htMinusVl << " found " << myGlobalEnergy->getInputHtVlMinusWheel().at(bx) << " bx "
           << bx << endl;
      testPass = false;
    }

    if (!myGlobalEnergy->getInputHtValPlusWheel().at(bx).overFlow() && !httPlusOverFlow &&
        (myGlobalEnergy->getInputHtValPlusWheel().at(bx).value() != htPlusVal)) {
      cout << "ht Plus " << htPlusVal << " bx " << bx << endl;
      testPass = false;
    }

    if (myGlobalEnergy->getInputHtValPlusWheel().at(bx).value() != htPlusVal) {
      cout << "ht Plus ovF " << htPlusVal << " found " << myGlobalEnergy->getInputHtValPlusWheel().at(bx) << " bx "
           << bx << endl;
      testPass = false;
    }

    if (myGlobalEnergy->getInputHxVlMinusWheel().at(bx).value() != hxMinusVl) {
      cout << "hx Minus " << hxMinusVl << " bx " << bx << endl;
      testPass = false;
    }
    if (myGlobalEnergy->getInputHxValPlusWheel().at(bx).value() != hxPlusVal) {
      cout << "hx Plus " << hxPlusVal << " bx " << bx << endl;
      testPass = false;
    }

    if (myGlobalEnergy->getInputHyVlMinusWheel().at(bx).value() != hyMinusVl) {
      cout << "hy Minus " << hyMinusVl << " bx " << bx << endl;
      testPass = false;
    }
    if (myGlobalEnergy->getInputHyValPlusWheel().at(bx).value() != hyPlusVal) {
      cout << "hy Plus " << hyPlusVal << " bx " << bx << endl;
      testPass = false;
    }

    if ((myGlobalEnergy->getInputHtVlMinusWheel().at(bx).overFlow() == httMinusOvrFlow) &&
        (myGlobalEnergy->getInputHxVlMinusWheel().at(bx).overFlow() == htmMinusOvrFlow) &&
        (myGlobalEnergy->getInputHyVlMinusWheel().at(bx).overFlow() == htmMinusOvrFlow)) {
    } else {
      cout << "Ht minus overflow check wheel"
           << " bx " << bx << endl;
      testPass = false;
    }

    if ((myGlobalEnergy->getInputHtValPlusWheel().at(bx).overFlow() == httPlusOverFlow) &&
        (myGlobalEnergy->getInputHxValPlusWheel().at(bx).overFlow() == htmPlusOverFlow) &&
        (myGlobalEnergy->getInputHyValPlusWheel().at(bx).overFlow() == htmPlusOverFlow)) {
    } else {
      cout << "Ht plus overflow check wheel"
           << " bx " << bx << endl;
      testPass = false;
    }

    // Check the output value
    if (!myGlobalEnergy->getEtHadColl().at(bx).overFlow() && !httTotalOvrFlow &&
        (myGlobalEnergy->getEtHadColl().at(bx).value() != htTotal)) {
      cout << "Algo etHad"
           << " bx " << bx << endl;
      testPass = false;
    }

    if (myGlobalEnergy->getEtHadColl().at(bx).value() != htTotal) {
      cout << "Algo etHad ovf"
           << " found " << myGlobalEnergy->getEtHadColl().at(bx) << " bx " << bx << endl;
      testPass = false;
    }

    // Check the missing Ht calculation
    unsigned htMiss = 0;
    unsigned htMPhi = 9;

    if ((htmMinusOvrFlow || htmPlusOverFlow) || ((abs(hxTotal) > 2047) || (abs(hyTotal) > 2047))) {
      hxTotal = 2047;
      hyTotal = 2047;
    }

    if ((((hxTotal)&0xff0) != 0) || (((hyTotal)&0xff0) != 0)) {
      double dhx = htComponentGeVForHtMiss(hxTotal);
      double dhy = htComponentGeVForHtMiss(hyTotal);
      double dhm = sqrt(dhx * dhx + dhy * dhy);
      double phi = atan2(dhy, dhx) + M_PI;

      htMiss = m_htMissScale->rank(dhm);
      htMPhi = static_cast<unsigned>(phi / M_PI * 9.);
    }

    if ((htMiss != myGlobalEnergy->getHtMissColl().at(bx).value()) ||
        (htMPhi != myGlobalEnergy->getHtMissPhiColl().at(bx).value())) {
      cout << "Missing Ht: expected " << htMiss << " phi " << htMPhi << " from x input " << hxTotal << " y input "
           << hyTotal << ", found " << myGlobalEnergy->getHtMissColl().at(bx).value() << " phi "
           << myGlobalEnergy->getHtMissPhiColl().at(bx).value() << " bx " << bx << endl;
      testPass = false;
    }

    // Check the storage of internal jet candidates
    unsigned htFromInternalJets = 0;
    bool htOvfFromInternalJets = false;
    for (L1GctInternJetDataCollection::const_iterator jet = internalJets.begin(); jet != internalJets.end(); jet++) {
      if ((jet->bx() == bx + m_bxStart) && (jet->et() >= m_jfPars->getHtJetEtThresholdGct())) {
        htFromInternalJets += jet->et();
        htOvfFromInternalJets |= jet->oflow();
      }
    }
    if ((htFromInternalJets >= 4096) || htOvfFromInternalJets)
      htFromInternalJets = 4095;
    if (htFromInternalJets != htTotal) {
      cout << "Found ht from jets " << htFromInternalJets << " expected " << htTotal << " bx " << bx << endl;
      testPass = false;
    }

    // end of loop over bunch crossings
  }
  return testPass;
}

//=================================================================================================================
//
// PRIVATE MEMBER FUNCTIONS
//
gctTestHt::rawJetData gctTestHt::rawJetFinderOutput(const L1GctJetFinderBase* jf,
                                                    const unsigned phiPos,
                                                    const int bx) const {
  assert(phiPos < 9);
  RawJetsVector jetsFromJf = jf->getRawJets();
  RawJetsVector jetList;
  unsigned sumHtt = 0;
  unsigned sumHtStrip0 = 0;
  unsigned sumHtStrip1 = 0;
  bool sumHttOvrFlo = false;
  bool sumHtmOvrFlo = false;
  for (RawJetsVector::const_iterator jet = jetsFromJf.begin(); jet != jetsFromJf.end(); jet++) {
    if (jet->bx() == bx && !jet->isNullJet()) {
      //       cout << "found a jet " << jet->rawsum()
      //   	   << " eta " << jet->globalEta()
      //   	   << " phi " << jet->globalPhi()
      // 	   << (jet->overFlow() ? " overflow set " : " ")
      // 	   << (jet->isTauJet() ? " tau jet " : " ")
      // 	   << (jet->isCentralJet() ? " central jet " : " ")
      // 	   << (jet->isForwardJet() ? " forward jet " : " ")
      // 	   << " bx " << jet->bx() << endl;
      jetList.push_back(*jet);
      // Find jet ht using event setup information
      //       double etJetGeV = jet->rawsum()*m_jetEtScale->linearLsb();
      //       double htJetGeV = m_jfPars->correctedEtGeV(etJetGeV, jet->rctEta(), jet->tauVeto());
      //       unsigned htJet  = ( jet->rawsum()==0x3ff ? 0x3ff : m_jfPars->correctedEtGct(htJetGeV));
      unsigned htJet = (jet->rawsum() == 0x3ff ? 0x3ff : jet->rawsum());
      // Add to total Ht sum
      if (htJet >= m_jfPars->getHtJetEtThresholdGct()) {
        sumHtt += htJet;
        sumHttOvrFlo |= (jet->overFlow());
      }
      // Add to missing Ht sum
      if (htJet >= m_jfPars->getMHtJetEtThresholdGct()) {
        if (jet->rctPhi() == 0) {
          sumHtStrip0 += htJet;
        }
        if (jet->rctPhi() == 1) {
          sumHtStrip1 += htJet;
        }
        sumHtmOvrFlo |= (jet->overFlow());
      }
    }
  }
  // Find the x and y components
  unsigned xFact0 = (53 - 4 * phiPos) % 36;
  unsigned xFact1 = (59 - 4 * phiPos) % 36;
  unsigned yFact0 = (44 - 4 * phiPos) % 36;
  unsigned yFact1 = (50 - 4 * phiPos) % 36;

  int sumHtx = htComponent(sumHtStrip0, xFact0, sumHtStrip1, xFact1);
  int sumHty = htComponent(sumHtStrip0, yFact0, sumHtStrip1, yFact1);

  // Check for overflow
  const int maxOutput = 0x800;
  while (sumHtx >= maxOutput) {
    sumHtx -= maxOutput * 2;
    sumHtmOvrFlo = true;
  }
  while (sumHtx < -maxOutput) {
    sumHtx += maxOutput * 2;
    sumHtmOvrFlo = true;
  }
  while (sumHty >= maxOutput) {
    sumHty -= maxOutput * 2;
    sumHtmOvrFlo = true;
  }
  while (sumHty < -maxOutput) {
    sumHty += maxOutput * 2;
    sumHtmOvrFlo = true;
  }

  rawJetData result(jetList, sumHtt, sumHtx, sumHty, sumHttOvrFlo, sumHtmOvrFlo);
  return result;
}

int gctTestHt::htComponent(const unsigned Emag0,
                           const unsigned fact0,
                           const unsigned Emag1,
                           const unsigned fact1) const {
  // Copy the Ex, Ey conversion from the hardware emulation
  const unsigned sinFact[10] = {0, 2845, 5603, 8192, 10531, 12550, 14188, 15395, 16134, 16383};
  unsigned myFact;
  bool neg0 = false, neg1 = false, negativeResult;
  int res0 = 0, res1 = 0, result;
  unsigned Emag, fact;

  for (int i = 0; i < 2; i++) {
    if (i == 0) {
      Emag = Emag0;
      fact = fact0;
    } else {
      Emag = Emag1;
      fact = fact1;
    }

    switch (fact / 9) {
      case 0:
        myFact = sinFact[fact];
        negativeResult = false;
        break;
      case 1:
        myFact = sinFact[(18 - fact)];
        negativeResult = false;
        break;
      case 2:
        myFact = sinFact[(fact - 18)];
        negativeResult = true;
        break;
      case 3:
        myFact = sinFact[(36 - fact)];
        negativeResult = true;
        break;
      default:
        cout << "Invalid factor " << fact << endl;
        return 0;
    }
    result = static_cast<int>(Emag * myFact);
    if (i == 0) {
      res0 = result;
      neg0 = negativeResult;
    } else {
      res1 = result;
      neg1 = negativeResult;
    }
  }
  if (neg0 == neg1) {
    result = res0 + res1;
    negativeResult = neg0;
  } else {
    if (res0 >= res1) {
      result = res0 - res1;
      negativeResult = neg0;
    } else {
      result = res1 - res0;
      negativeResult = neg1;
    }
  }
  // Divide by 8192 using bit-shift; but emulate
  // twos-complement arithmetic for negative numbers
  if (negativeResult) {
    result = (1 << 28) - result;
    result = (result + 0x1000) >> 13;
    result = result - (1 << 15);
  } else {
    result = (result + 0x1000) >> 13;
  }

  return result;
}

double gctTestHt::htComponentGeVForHtMiss(int inputComponent) const {
  // Deal properly with the LSB truncation for 2s-complement numbers
  // Input is 18 bits including sign bit but we use only 8 bits
  int truncatedComponent = (((inputComponent + 0x800) >> 4) & 0xff) - 0x80;
  return ((static_cast<double>(truncatedComponent) + 0.5) * 8.0 * m_jfPars->getHtLsbGeV());
}
