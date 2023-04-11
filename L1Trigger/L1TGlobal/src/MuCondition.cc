/**
 * \class MuCondition
 *
 *
 * Description: evaluation of a CondMuon condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *          Vladimir Rekovic - extend for indexing
 *          Rick Cavanaugh - include displaced muons
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/MuCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::MuCondition::MuCondition() : ConditionEvaluation() {
  // empty
}

//     from base template condition (from event setup usually)
l1t::MuCondition::MuCondition(const GlobalCondition* muonTemplate,
                              const GlobalBoard* ptrGTL,
                              const int nrL1Mu,
                              const int ifMuEtaNumberBits)
    : ConditionEvaluation(),
      m_gtMuonTemplate(static_cast<const MuonTemplate*>(muonTemplate)),
      m_gtGTL(ptrGTL),
      m_ifMuEtaNumberBits(ifMuEtaNumberBits) {
  m_corrParDeltaPhiNrBins = 0;
  m_condMaxNumberObjects = nrL1Mu;
}

// copy constructor
void l1t::MuCondition::copy(const l1t::MuCondition& cp) {
  m_gtMuonTemplate = cp.gtMuonTemplate();
  m_gtGTL = cp.gtGTL();

  m_ifMuEtaNumberBits = cp.gtIfMuEtaNumberBits();
  m_corrParDeltaPhiNrBins = cp.m_corrParDeltaPhiNrBins;

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::MuCondition::MuCondition(const l1t::MuCondition& cp) : ConditionEvaluation() { copy(cp); }

// destructor
l1t::MuCondition::~MuCondition() {
  // empty
}

// equal operator
l1t::MuCondition& l1t::MuCondition::operator=(const l1t::MuCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::MuCondition::setGtMuonTemplate(const MuonTemplate* muonTempl) { m_gtMuonTemplate = muonTempl; }

///   set the pointer to GTL
void l1t::MuCondition::setGtGTL(const GlobalBoard* ptrGTL) { m_gtGTL = ptrGTL; }

//   set the number of bits for eta of muon objects
void l1t::MuCondition::setGtIfMuEtaNumberBits(const int& ifMuEtaNumberBitsValue) {
  m_ifMuEtaNumberBits = ifMuEtaNumberBitsValue;
}

//   set the maximum number of bins for the delta phi scales
void l1t::MuCondition::setGtCorrParDeltaPhiNrBins(const int& corrParDeltaPhiNrBins) {
  m_corrParDeltaPhiNrBins = corrParDeltaPhiNrBins;
}

// try all object permutations and check spatial correlations, if required
const bool l1t::MuCondition::evaluateCondition(const int bxEval) const {
  // BLW Need to pass this as an argument
  //const int bxEval=0;   //BLW Change for BXVector

  // number of trigger objects in the condition
  int nObjInCond = m_gtMuonTemplate->nrObjects();

  // the candidates
  const BXVector<const l1t::Muon*>* candVec = m_gtGTL->getCandL1Mu();  //BLW Change for BXVector

  // Look at objects in bx = bx + relativeBx
  int useBx = bxEval + m_gtMuonTemplate->condRelativeBx();

  // Fail condition if attempting to get Bx outside of range
  if ((useBx < candVec->getFirstBX()) || (useBx > candVec->getLastBX())) {
    return false;
  }

  int numberObjects = candVec->size(useBx);  //BLW Change for BXVector
  //LogTrace("L1TGlobal") << "  numberObjects: " << numberObjects
  //    << std::endl;
  if (numberObjects < nObjInCond) {
    return false;
  }

  std::vector<int> index(numberObjects);

  for (int i = 0; i < numberObjects; ++i) {
    index[i] = i;
  }

  int numberForFactorial = numberObjects - nObjInCond;

  // TEMPORARY FIX UNTIL IMPLEMENT NEW MUON CONDITIONS
  int myfactorial = 1;
  for (int i = numberForFactorial; i > 0; i--)
    myfactorial *= i;

  int jumpIndex = 1;
  int jump = myfactorial;  //factorial(numberObjects - nObjInCond);

  int totalLoops = 0;
  int passLoops = 0;

  // condition result condResult set to true if at least one permutation
  //     passes all requirements
  // all possible permutations are checked
  bool condResult = false;

  // store the indices of the muon objects
  // from the combination evaluated in the condition
  SingleCombInCond objectsInComb;
  objectsInComb.reserve(nObjInCond);

  // clear the m_combinationsInCond vector
  combinationsInCond().clear();

  do {
    if (--jumpIndex)
      continue;

    jumpIndex = jump;
    totalLoops++;

    // clear the indices in the combination
    objectsInComb.clear();

    bool tmpResult = true;

    bool passCondition = false;
    // check if there is a permutation that matches object-parameter requirements
    for (int i = 0; i < nObjInCond; i++) {
      passCondition = checkObjectParameter(i, *(candVec->at(useBx, index[i])), index[i]);  //BLW Change for BXVector
      tmpResult &= passCondition;
      if (passCondition)
        LogDebug("L1TGlobal") << "===> MuCondition::evaluateCondition, CONGRATS!! This muon passed the condition."
                              << std::endl;
      else
        LogDebug("L1TGlobal") << "===> MuCondition::evaluateCondition, FAIL!! This muon failed the condition."
                              << std::endl;
      objectsInComb.push_back(index[i]);
    }

    // if permutation does not match particle conditions
    // skip charge correlation and spatial correlations
    if (!tmpResult) {
      continue;
    }

    // get the correlation parameters (chargeCorrelation included here also)
    MuonTemplate::CorrelationParameter corrPar = *(m_gtMuonTemplate->correlationParameter());

    // charge_correlation consists of 3 relevant bits (D2, D1, D0)
    unsigned int chargeCorr = corrPar.chargeCorrelation;

    // charge ignore bit (D0) not set?
    if ((chargeCorr & 1) == 0) {
      LogDebug("L1TGlobal") << "===> MuCondition:: Checking Charge Correlation" << std::endl;

      for (int i = 0; i < nObjInCond; i++) {
        // check valid charge - skip if invalid charge
        int chargeValid = (candVec->at(useBx, index[i]))->hwChargeValid();  //BLW Change for BXVector
        tmpResult &= chargeValid;

        if (chargeValid == 0) {  //BLW type change for New Muon Class
          continue;
        }
      }

      if (!tmpResult) {
        LogDebug("L1TGlobal") << "===> MuCondition:: Charge Correlation Failed...No Valid Charges" << std::endl;
        continue;
      }

      if (nObjInCond > 1) {  // more objects condition

        // find out if signs are equal
        bool equalSigns = true;
        for (int i = 0; i < nObjInCond - 1; i++) {
          if ((candVec->at(useBx, index[i]))->hwCharge() !=
              (candVec->at(useBx, index[i + 1]))->hwCharge()) {  //BLW Change for BXVector
            equalSigns = false;
            break;
          }
        }

        LogDebug("L1TGlobal") << "===> MuCondition:: Checking Charge Correlation equalSigns = " << equalSigns
                              << std::endl;

        // two or three particle condition
        if (nObjInCond == 2 || nObjInCond == 3) {
          if (!(((chargeCorr & 2) != 0 && equalSigns) || ((chargeCorr & 4) != 0 && !equalSigns))) {
            LogDebug("L1TGlobal") << "===> MuCondition:: 2/3 Muon Fail Charge Correlation Condition =" << chargeCorr
                                  << std::endl;
            continue;
          }
        } else if (nObjInCond == 4) {
          //counter to count positive charges to determine if there are pairs
          unsigned int posCount = 0;

          for (int i = 0; i < nObjInCond; i++) {
            if ((candVec->at(useBx, index[i]))->hwCharge() > 0) {  //BLW Change for BXVector
              posCount++;
            }
          }

          // Original OS 4 muon condition (disagreement with firmware):
          //    if (!(((chargeCorr & 2) != 0 && equalSigns) || ((chargeCorr & 4) != 0 && posCount == 2))) {
          // Fix by R. Cavanaugh:
          //       Note that negative charge => hwCharge = 0
          //                 positive charge => hwCharge = 1
          //       Hence:  (0,0,0,0) => (posCount = 0) => 4 SS muons
          //               (1,0,0,0) => (posCount = 1) => 1 OS muon pair, 1 SS muon pair
          //               (1,1,0,0) => (posCount = 2) => 2 OS muon pairs
          //               (1,0,1,0) => (posCount = 2) => 2 OS muon pairs
          //               (0,0,1,1) => (posCount = 2) => 2 OS muon pairs
          //               (1,1,1,0) => (posCount = 3) => 1 SS muon pair, 1 OS muon pair
          //               (1,1,1,1) => (posCount = 4) => 4 SS muons
          //       A requirement (posCount == 2) implies there must be exactly 2 OS pairs of muons
          //       A requirement of at least 1 pair of OS muons implies condition should be (posCount > 0 && posCount < 4)
          if (!(((chargeCorr & 2) != 0 && equalSigns) || ((chargeCorr & 4) != 0 && (posCount > 0 && posCount < 4)))) {
            LogDebug("L1TGlobal") << "===> MuCondition:: 4 Muon Fail Charge Correlation Condition = " << chargeCorr
                                  << " posCnt " << posCount << std::endl;
            continue;
          }
        }
      }  // end require nObjInCond > 1
    }    // end signchecks

    if (m_gtMuonTemplate->wsc()) {
      // wsc requirements have always nObjInCond = 2
      // one can use directly index[0] and index[1] to compute
      // eta and phi differences
      const int ObjInWscComb = 2;
      if (nObjInCond != ObjInWscComb) {
        edm::LogError("L1TGlobal") << "\n  Error: "
                                   << "number of particles in condition with spatial correlation = " << nObjInCond
                                   << "\n  it must be = " << ObjInWscComb << std::endl;
        // TODO Perhaps I should throw here an exception,
        // since something is really wrong if nObjInCond != ObjInWscComb (=2)
        continue;
      }

      // check delta eta
      if (!checkRangeDeltaEta((candVec->at(useBx, 0))->hwEtaAtVtx(),
                              (candVec->at(useBx, 1))->hwEtaAtVtx(),
                              corrPar.deltaEtaRangeLower,
                              corrPar.deltaEtaRangeUpper,
                              8)) {
        LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed checkRangeDeltaEta" << std::endl;
        continue;
      }

      // check delta phi
      if (!checkRangeDeltaPhi((candVec->at(useBx, 0))->hwPhiAtVtx(),
                              (candVec->at(useBx, 1))->hwPhiAtVtx(),
                              corrPar.deltaPhiRangeLower,
                              corrPar.deltaPhiRangeUpper)) {
        LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed checkRangeDeltaPhi" << std::endl;
        continue;
      }

    }  // end wsc check

    // if we get here all checks were successfull for this combination
    // set the general result for evaluateCondition to "true"

    condResult = true;
    passLoops++;
    (combinationsInCond()).push_back(objectsInComb);

  } while (std::next_permutation(index.begin(), index.end()));

  //LogTrace("L1TGlobal")
  //    << "\n  MuCondition: total number of permutations found:          " << totalLoops
  //    << "\n  MuCondition: number of permutations passing requirements: " << passLoops
  //    << "\n" << std::endl;

  return condResult;
}

// load muon candidates
const l1t::Muon* l1t::MuCondition::getCandidate(const int bx, const int indexCand) const {
  return (m_gtGTL->getCandL1Mu())->at(bx, indexCand);  //BLW Change for BXVector
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition.
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool l1t::MuCondition::checkObjectParameter(const int iCondition,
                                                  const l1t::Muon& cand,
                                                  const unsigned int index) const {
  // number of objects in condition
  int nObjInCond = m_gtMuonTemplate->nrObjects();

  if (iCondition >= nObjInCond || iCondition < 0) {
    return false;
  }

  //     // empty candidates can not be compared
  //     if (cand.empty()) {
  //         return false;
  //     }

  const MuonTemplate::ObjectParameter objPar = (*(m_gtMuonTemplate->objectParameter()))[iCondition];

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

  LogDebug("L1TGlobal") << "\n MuonTemplate::ObjectParameter : " << std::hex << "\n\t ptHighThreshold = 0x "
                        << objPar.ptHighThreshold << "\n\t ptLowThreshold  = 0x " << objPar.ptLowThreshold
                        << "\n\t indexHigh       = 0x " << objPar.indexHigh << "\n\t indexLow        = 0x "
                        << objPar.indexLow << "\n\t requestIso      = 0x " << objPar.requestIso
                        << "\n\t enableIso       = 0x " << objPar.enableIso << "\n\t etaRange        = 0x "
                        << objPar.etaRange << "\n\t phiLow          = 0x " << objPar.phiLow
                        << "\n\t phiHigh         = 0x " << objPar.phiHigh << "\n\t phiWindow1Lower = 0x "
                        << objPar.phiWindow1Lower << "\n\t phiWindow1Upper = 0x " << objPar.phiWindow1Upper
                        << "\n\t phiWindow2Lower = 0x " << objPar.phiWindow2Lower << "\n\t phiWindow2Lower = 0x "
                        << objPar.phiWindow2Lower << "\n\t charge          = 0x " << objPar.charge
                        << "\n\t qualityLUT      = 0x " << objPar.qualityLUT << "\n\t isolationLUT    = 0x "
                        << objPar.isolationLUT << "\n\t enableMip       = 0x " << objPar.enableMip << std::endl;

  LogDebug("L1TGlobal") << "\n l1t::Muon : "
                        << "\n\t hwPt       = 0x " << cand.hwPt() << "\n\t hwEtaAtVtx = 0x " << cand.hwEtaAtVtx()
                        << "\n\t hwPhiAtVtx = 0x " << cand.hwPhiAtVtx() << "\n\t hwCharge   = 0x " << cand.hwCharge()
                        << "\n\t hwQual     = 0x " << cand.hwQual() << "\n\t hwIso      = 0x " << cand.hwIso()
                        << std::dec << std::endl;

  if (objPar.unconstrainedPtHigh > 0)  // Rick Cavanaugh:  Check if unconstrained pT cut-window is valid
  {
    if (!checkUnconstrainedPt(objPar.unconstrainedPtLow,
                              objPar.unconstrainedPtHigh,
                              cand.hwPtUnconstrained(),
                              m_gtMuonTemplate->condGEq())) {
      LogDebug("L1TGlobal") << "\t\t Muon Failed unconstrainedPt checkThreshold; iCondition = " << iCondition
                            << std::endl;
      return false;
    }
  }
  if (objPar.impactParameterLUT !=
      0)  // Rick Cavanaugh:  Check if impact parameter LUT is valid.  0xF is default; 0x0 is invalid
  {
    // check impact parameter ( bit check ) with impact parameter LUT
    // sanity check on candidate impact parameter
    if (cand.hwDXY() > 3) {
      LogDebug("L1TGlobal") << "\t\t l1t::Candidate has out of range hwDXY = " << cand.hwDXY() << std::endl;
      return false;
    }
    bool passImpactParameterLUT = ((objPar.impactParameterLUT >> cand.hwDXY()) & 1);
    if (!passImpactParameterLUT) {
      LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed impact parameter requirement" << std::endl;
      return false;
    }
  }

  if (!checkThreshold(objPar.ptLowThreshold, objPar.ptHighThreshold, cand.hwPt(), m_gtMuonTemplate->condGEq())) {
    LogDebug("L1TGlobal") << "\t\t Muon Failed checkThreshold " << std::endl;
    return false;
  }

  // check index
  if (!checkIndex(objPar.indexLow, objPar.indexHigh, index)) {
    LogDebug("L1TGlobal") << "\t\t Muon Failed checkIndex " << std::endl;
    return false;
  }

  // check eta
  if (!checkRangeEta(cand.hwEtaAtVtx(),
                     objPar.etaWindow1Lower,
                     objPar.etaWindow1Upper,
                     objPar.etaWindow2Lower,
                     objPar.etaWindow2Upper,
                     objPar.etaWindow3Lower,
                     objPar.etaWindow3Upper,
                     8)) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed checkRange(eta)" << std::endl;
    return false;
  }

  // check phi
  if (!checkRangePhi(cand.hwPhiAtVtx(),
                     objPar.phiWindow1Lower,
                     objPar.phiWindow1Upper,
                     objPar.phiWindow2Lower,
                     objPar.phiWindow2Upper)) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed checkRange(phi)" << std::endl;
    return false;
  }

  // check charge
  if (objPar.charge >= 0) {
    if (cand.hwCharge() != objPar.charge) {
      LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed charge requirement" << std::endl;
      return false;
    }
  }

  // check quality ( bit check ) with quality LUT
  // sanity check on candidate quality
  if (cand.hwQual() > 16) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate has out of range hwQual = " << cand.hwQual() << std::endl;
    return false;
  }
  bool passQualLUT = ((objPar.qualityLUT >> cand.hwQual()) & 1);
  if (!passQualLUT) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed quality requirement" << std::endl;
    return false;
  }

  // check isolation ( bit check ) with isolation LUT
  // sanity check on candidate isolation
  if (cand.hwIso() > 4) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate has out of range hwIso = " << cand.hwIso() << std::endl;
    return false;
  }
  bool passIsoLUT = ((objPar.isolationLUT >> cand.hwIso()) & 1);
  if (!passIsoLUT) {
    LogDebug("L1TGlobal") << "\t\t l1t::Candidate failed isolation requirement" << std::endl;
    return false;
  }

  // A number of values is required to trigger (at least one).
  // "Don't care" means that all values are allowed.
  // Qual = 000 means then NO MUON (GTL module)

  // if (cand.hwQual() == 0) {
  // 	LogDebug("L1TGlobal") << "\t\t Muon Failed hwQual() == 0" << std::endl;
  //     return false;
  // }

  // if (objPar.qualityRange == 0) {
  // 	LogDebug("L1TGlobal") << "\t\t Muon Failed qualityRange == 0" << std::endl;
  //     return false;
  // }
  // else {
  //   if (!checkBit(objPar.qualityRange, cand.hwQual())) {
  // 	LogDebug("L1TGlobal") << "\t\t Muon Failed checkBit(qualityRange) " << std::endl;
  //         return false;
  //     }
  // }

  // check mip
  if (objPar.enableMip) {
    //      if (!cand.hwMip()) {
    // LogDebug("L1TGlobal") << "\t\t Muon Failed enableMip" << std::endl;
    //          return false;
    //      }
  }

  // particle matches if we get here
  //LogTrace("L1TGlobal")
  //    << "  checkObjectParameter: muon object OK, passes all requirements\n" << std::endl;

  return true;
}

void l1t::MuCondition::print(std::ostream& myCout) const {
  m_gtMuonTemplate->print(myCout);

  myCout << "    Number of bits for eta of muon objects = " << m_ifMuEtaNumberBits << std::endl;
  myCout << "    Maximum number of bins for the delta phi scales = " << m_corrParDeltaPhiNrBins << "\n " << std::endl;

  ConditionEvaluation::print(myCout);
}
