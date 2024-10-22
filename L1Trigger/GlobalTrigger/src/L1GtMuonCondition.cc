/**
 * \class L1GtMuonCondition
 *
 *
 * Description: evaluation of a CondMuon condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtMuonCondition.h"

// system include files
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <string>
#include <vector>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
//     default
L1GtMuonCondition::L1GtMuonCondition() : L1GtConditionEvaluation() {
  // empty
}

//     from base template condition (from event setup usually)
L1GtMuonCondition::L1GtMuonCondition(const L1GtCondition *muonTemplate,
                                     const L1GlobalTriggerGTL *ptrGTL,
                                     const int nrL1Mu,
                                     const int ifMuEtaNumberBits)
    : L1GtConditionEvaluation(),
      m_gtMuonTemplate(static_cast<const L1GtMuonTemplate *>(muonTemplate)),
      m_gtGTL(ptrGTL),
      m_ifMuEtaNumberBits(ifMuEtaNumberBits) {
  m_corrParDeltaPhiNrBins = 0;
  m_condMaxNumberObjects = nrL1Mu;
}

// copy constructor
void L1GtMuonCondition::copy(const L1GtMuonCondition &cp) {
  m_gtMuonTemplate = cp.gtMuonTemplate();
  m_gtGTL = cp.gtGTL();

  m_ifMuEtaNumberBits = cp.gtIfMuEtaNumberBits();
  m_corrParDeltaPhiNrBins = cp.m_corrParDeltaPhiNrBins;

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

L1GtMuonCondition::L1GtMuonCondition(const L1GtMuonCondition &cp) : L1GtConditionEvaluation() { copy(cp); }

// destructor
L1GtMuonCondition::~L1GtMuonCondition() {
  // empty
}

// equal operator
L1GtMuonCondition &L1GtMuonCondition::operator=(const L1GtMuonCondition &cp) {
  copy(cp);
  return *this;
}

// methods
void L1GtMuonCondition::setGtMuonTemplate(const L1GtMuonTemplate *muonTempl) { m_gtMuonTemplate = muonTempl; }

///   set the pointer to GTL
void L1GtMuonCondition::setGtGTL(const L1GlobalTriggerGTL *ptrGTL) { m_gtGTL = ptrGTL; }

//   set the number of bits for eta of muon objects
void L1GtMuonCondition::setGtIfMuEtaNumberBits(const int &ifMuEtaNumberBitsValue) {
  m_ifMuEtaNumberBits = ifMuEtaNumberBitsValue;
}

//   set the maximum number of bins for the delta phi scales
void L1GtMuonCondition::setGtCorrParDeltaPhiNrBins(const int &corrParDeltaPhiNrBins) {
  m_corrParDeltaPhiNrBins = corrParDeltaPhiNrBins;
}

// try all object permutations and check spatial correlations, if required
const bool L1GtMuonCondition::evaluateCondition() const {
  // number of trigger objects in the condition
  int nObjInCond = m_gtMuonTemplate->nrObjects();

  // the candidates
  const std::vector<const L1MuGMTCand *> *candVec = m_gtGTL->getCandL1Mu();

  int numberObjects = candVec->size();
  // LogTrace("L1GlobalTrigger") << "  numberObjects: " << numberObjects
  //    << std::endl;
  if (numberObjects < nObjInCond) {
    return false;
  }

  std::vector<int> index(numberObjects);

  for (int i = 0; i < numberObjects; ++i) {
    index[i] = i;
  }

  int jumpIndex = 1;
  int jump = factorial(numberObjects - nObjInCond);

  // condition result condResult set to true if at least one permutation
  //     passes all requirements
  // all possible permutations are checked
  bool condResult = false;

  // store the indices of the muon objects
  // from the combination evaluated in the condition
  SingleCombInCond objectsInComb;
  objectsInComb.reserve(nObjInCond);

  // clear the m_combinationsInCond vector
  (combinationsInCond()).clear();

  do {
    if (--jumpIndex)
      continue;

    jumpIndex = jump;

    // clear the indices in the combination
    objectsInComb.clear();

    bool tmpResult = true;

    // check if there is a permutation that matches object-parameter
    // requirements
    for (int i = 0; i < nObjInCond; i++) {
      tmpResult &= checkObjectParameter(i, *(*candVec)[index[i]]);
      objectsInComb.push_back(index[i]);
    }

    // if permutation does not match particle conditions
    // skip charge correlation and spatial correlations
    if (!tmpResult) {
      continue;
    }

    // get the correlation parameters (chargeCorrelation included here also)
    L1GtMuonTemplate::CorrelationParameter corrPar = *(m_gtMuonTemplate->correlationParameter());

    // charge_correlation consists of 3 relevant bits (D2, D1, D0)
    unsigned int chargeCorr = corrPar.chargeCorrelation;

    // charge ignore bit (D0) not set?
    if ((chargeCorr & 1) == 0) {
      for (int i = 0; i < nObjInCond; i++) {
        // check valid charge - skip if invalid charge
        bool chargeValid = (*candVec)[index[i]]->charge_valid();
        tmpResult &= chargeValid;

        if (!chargeValid) {
          continue;
        }
      }

      if (!tmpResult) {
        continue;
      }

      if (nObjInCond == 1) {  // one object condition

        // D2..enable pos, D1..enable neg
        if (!(((chargeCorr & 4) != 0 && (*candVec)[index[0]]->charge() > 0) ||
              ((chargeCorr & 2) != 0 && (*candVec)[index[0]]->charge() < 0))) {
          continue;
        }

      } else {  // more objects condition

        // find out if signs are equal
        bool equalSigns = true;
        for (int i = 0; i < nObjInCond - 1; i++) {
          if ((*candVec)[index[i]]->charge() != (*candVec)[index[i + 1]]->charge()) {
            equalSigns = false;
            break;
          }
        }

        // two or three particle condition
        if (nObjInCond == 2 || nObjInCond == 3) {
          // D2..enable equal, D1..enable not equal
          if (!(((chargeCorr & 4) != 0 && equalSigns) || ((chargeCorr & 2) != 0 && !equalSigns))) {
            continue;
          }
        }

        // four particle condition
        if (nObjInCond == 4) {
          // counter to count positive charges to determine if there are pairs
          unsigned int posCount = 0;

          for (int i = 0; i < nObjInCond; i++) {
            if ((*candVec)[index[i]]->charge() > 0) {
              posCount++;
            }
          }

          // D2..enable equal, D1..enable pairs
          if (!(((chargeCorr & 4) != 0 && equalSigns) || ((chargeCorr & 2) != 0 && posCount == 2))) {
            continue;
          }
        }
      }
    }  // end signchecks

    if (m_gtMuonTemplate->wsc()) {
      // wsc requirements have always nObjInCond = 2
      // one can use directly index[0] and index[1] to compute
      // eta and phi differences
      const int ObjInWscComb = 2;
      if (nObjInCond != ObjInWscComb) {
        edm::LogError("L1GlobalTrigger") << "\n  Error: "
                                         << "number of particles in condition with spatial correlation = " << nObjInCond
                                         << "\n  it must be = " << ObjInWscComb << std::endl;
        // TODO Perhaps I should throw here an exception,
        // since something is really wrong if nObjInCond != ObjInWscComb (=2)
        continue;
      }

      unsigned int candDeltaEta;
      unsigned int candDeltaPhi;

      // check candDeltaEta

      // get eta index and the sign bit of the eta index (MSB is the sign)
      //   signedEta[i] is the signed eta index of (*candVec)[index[i]]
      int signedEta[ObjInWscComb];
      int signBit[ObjInWscComb] = {0, 0};

      int scaleEta = 1 << (m_ifMuEtaNumberBits - 1);

      for (int i = 0; i < ObjInWscComb; ++i) {
        signBit[i] = ((*candVec)[index[i]]->etaIndex() & scaleEta) >> (m_ifMuEtaNumberBits - 1);
        signedEta[i] = ((*candVec)[index[i]]->etaIndex()) % scaleEta;

        if (signBit[i] == 1) {
          signedEta[i] = (-1) * signedEta[i];
        }
      }

      // compute candDeltaEta - add 1 if signs are different (due to +0/-0
      // indices)
      candDeltaEta =
          static_cast<int>(std::abs(signedEta[1] - signedEta[0])) + static_cast<int>(signBit[1] ^ signBit[0]);

      if (!checkBit(corrPar.deltaEtaRange, candDeltaEta)) {
        continue;
      }

      // check candDeltaPhi

      // calculate absolute value of candDeltaPhi
      if ((*candVec)[index[0]]->phiIndex() > (*candVec)[index[1]]->phiIndex()) {
        candDeltaPhi = (*candVec)[index[0]]->phiIndex() - (*candVec)[index[1]]->phiIndex();
      } else {
        candDeltaPhi = (*candVec)[index[1]]->phiIndex() - (*candVec)[index[0]]->phiIndex();
      }

      // check if candDeltaPhi > 180 (via delta_phi_maxbits)
      // delta_phi contains bits for 0..180 (0 and 180 included)
      // protect also against infinite loop...

      int nMaxLoop = 10;
      int iLoop = 0;

      while (candDeltaPhi >= m_corrParDeltaPhiNrBins) {
        unsigned int candDeltaPhiInitial = candDeltaPhi;

        // candDeltaPhi > 180 ==> take 360 - candDeltaPhi
        candDeltaPhi = (m_corrParDeltaPhiNrBins - 1) * 2 - candDeltaPhi;
        if (m_verbosity) {
          LogTrace("L1GlobalTrigger") << "    Initial candDeltaPhi = " << candDeltaPhiInitial
                                      << " > m_corrParDeltaPhiNrBins = " << m_corrParDeltaPhiNrBins
                                      << "  ==> candDeltaPhi rescaled to: " << candDeltaPhi << " [ loop index " << iLoop
                                      << "; breaks after " << nMaxLoop << " loops ]\n"
                                      << std::endl;
        }

        iLoop++;
        if (iLoop > nMaxLoop) {
          return false;
        }
      }

      // delta_phi bitmask is saved in two uint64_t words
      if (candDeltaPhi < 64) {
        if (!checkBit(corrPar.deltaPhiRange0Word, candDeltaPhi)) {
          continue;
        }
      } else {
        if (!checkBit(corrPar.deltaPhiRange1Word, (candDeltaPhi - 64))) {
          continue;
        }
      }

    }  // end wsc check

    // if we get here all checks were successfull for this combination
    // set the general result for evaluateCondition to "true"

    condResult = true;
    (combinationsInCond()).push_back(objectsInComb);

  } while (std::next_permutation(index.begin(), index.end()));
  return condResult;
}

// load muon candidates
const L1MuGMTCand *L1GtMuonCondition::getCandidate(const int indexCand) const {
  return (*(m_gtGTL->getCandL1Mu()))[indexCand];
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition.
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool L1GtMuonCondition::checkObjectParameter(const int iCondition, const L1MuGMTCand &cand) const {
  // number of objects in condition
  int nObjInCond = m_gtMuonTemplate->nrObjects();

  if (iCondition >= nObjInCond || iCondition < 0) {
    return false;
  }

  // empty candidates can not be compared
  if (cand.empty()) {
    return false;
  }

  const L1GtMuonTemplate::ObjectParameter objPar = (*(m_gtMuonTemplate->objectParameter()))[iCondition];

  // using the logic table from GTL-9U-module.pdf
  // "Truth table for Isolation bit"

  // check thresholds:

  //   value < low pt threshold
  //       fail trigger

  //   low pt threshold <= value < high pt threshold & non-isolated muon:
  //       requestIso true:                    fail trigger
  //       requestIso false, enableIso true:   fail trigger
  //       requestIso false, enableIso false:  OK,  trigger

  //   low pt threshold <= value < high pt threshold & isolated muon:
  //       requestIso true:                    OK,  trigger
  //       requestIso false, enableIso true:   OK,  trigger
  //       requestIso false, enableIso false:  OK,  trigger

  //   value >= high pt threshold & non-isolated muon:
  //       requestIso true:  fail trigger
  //       requestIso false: OK,  trigger

  //   value >= high pt threshold & isolated muon:
  //       OK, trigger

  if (!checkThreshold(objPar.ptHighThreshold, cand.ptIndex(), m_gtMuonTemplate->condGEq())) {
    if (!checkThreshold(objPar.ptLowThreshold, cand.ptIndex(), m_gtMuonTemplate->condGEq())) {
      return false;
    } else {
      // check isolation
      if (!cand.isol()) {
        if (objPar.requestIso || objPar.enableIso) {
          return false;
        }
      }
    }

  } else {
    if (!cand.isol()) {
      if (objPar.requestIso) {
        return false;
      }
    }
  }

  // check eta

  if (!checkBit(objPar.etaRange, cand.etaIndex())) {
    return false;
  }

  // check phi  - in the requested range (no LUT used - LUT too big for hw chip)
  // for phiLow <= phiHigh takes [phiLow, phiHigh]
  // for phiLow >= phiHigh takes [phiLow, phiHigh] over zero angle!

  if (objPar.phiHigh >= objPar.phiLow) {
    if (!((objPar.phiLow <= cand.phiIndex()) && (cand.phiIndex() <= objPar.phiHigh))) {
      return false;
    }

  } else {  // go over zero angle!!
    if (!((objPar.phiLow <= cand.phiIndex()) || (cand.phiIndex() <= objPar.phiHigh))) {
      return false;
    }
  }

  // check quality ( bit check )

  // A number of values is required to trigger (at least one).
  // "Don’t care" means that all values are allowed.
  // Qual = 000 means then NO MUON (GTL module)

  if (cand.quality() == 0) {
    return false;
  }

  if (objPar.qualityRange == 0) {
    return false;
  } else {
    if (!checkBit(objPar.qualityRange, cand.quality())) {
      return false;
    }
  }

  // check mip
  if (objPar.enableMip) {
    if (!cand.mip()) {
      return false;
    }
  }

  // particle matches if we get here
  // LogTrace("L1GlobalTrigger")
  //    << "  checkObjectParameter: muon object OK, passes all requirements\n"
  //    << std::endl;

  return true;
}

void L1GtMuonCondition::print(std::ostream &myCout) const {
  m_gtMuonTemplate->print(myCout);

  myCout << "    Number of bits for eta of muon objects = " << m_ifMuEtaNumberBits << std::endl;
  myCout << "    Maximum number of bins for the delta phi scales = " << m_corrParDeltaPhiNrBins << "\n " << std::endl;

  L1GtConditionEvaluation::print(myCout);
}
