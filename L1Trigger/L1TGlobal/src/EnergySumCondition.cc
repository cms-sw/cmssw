/**
 * \class EnergySumCondition
 *
 *
 * Description: evaluation of a CondEnergySum condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/EnergySumCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::EnergySumCondition::EnergySumCondition() : ConditionEvaluation() {
  //empty
}

//     from base template condition (from event setup usually)
l1t::EnergySumCondition::EnergySumCondition(const GlobalCondition* eSumTemplate, const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtEnergySumTemplate(static_cast<const EnergySumTemplate*>(eSumTemplate)),
      m_uGtB(ptrGTB)

{
  // maximum number of objects received for the evaluation of the condition
  // energy sums are global quantities - one object per event

  m_condMaxNumberObjects = 1;
}

// copy constructor
void l1t::EnergySumCondition::copy(const l1t::EnergySumCondition& cp) {
  m_gtEnergySumTemplate = cp.gtEnergySumTemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::EnergySumCondition::EnergySumCondition(const l1t::EnergySumCondition& cp) : ConditionEvaluation() { copy(cp); }

// destructor
l1t::EnergySumCondition::~EnergySumCondition() {
  // empty
}

// equal operator
l1t::EnergySumCondition& l1t::EnergySumCondition::operator=(const l1t::EnergySumCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::EnergySumCondition::setGtEnergySumTemplate(const EnergySumTemplate* eSumTempl) {
  m_gtEnergySumTemplate = eSumTempl;
}

///   set the pointer to uGT GlobalBoard
void l1t::EnergySumCondition::setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

// try all object permutations and check spatial correlations, if required
const bool l1t::EnergySumCondition::evaluateCondition(const int bxEval) const {
  // number of trigger objects in the condition
  // in fact, there is only one object
  int iCondition = 0;

  // condition result condResult set to true if the energy sum
  // passes all requirements
  bool condResult = false;

  // store the indices of the calorimeter objects
  // from the combination evaluated in the condition
  SingleCombInCond objectsInComb;

  // clear the m_combinationsInCond vector
  (combinationsInCond()).clear();

  // clear the indices in the combination
  objectsInComb.clear();

  const BXVector<const l1t::EtSum*>* candVec = m_uGtB->getCandL1EtSum();

  // Look at objects in bx = bx + relativeBx
  int useBx = bxEval + m_gtEnergySumTemplate->condRelativeBx();

  // Fail condition if attempting to get Bx outside of range
  if ((useBx < candVec->getFirstBX()) || (useBx > candVec->getLastBX())) {
    return false;
  }

  // If no candidates, no use looking any further.
  int numberObjects = candVec->size(useBx);
  if (numberObjects < 1) {
    return false;
  }

  l1t::EtSum::EtSumType type;
  bool MissingEnergy = false;
  int centbit = 0;
  switch ((m_gtEnergySumTemplate->objectType())[0]) {
    case gtETM:
      type = l1t::EtSum::EtSumType::kMissingEt;
      MissingEnergy = true;
      break;
    case gtETT:
      type = l1t::EtSum::EtSumType::kTotalEt;
      MissingEnergy = false;
      break;
    case gtETTem:
      type = l1t::EtSum::EtSumType::kTotalEtEm;
      MissingEnergy = false;
      break;
    case gtHTM:
      type = l1t::EtSum::EtSumType::kMissingHt;
      MissingEnergy = true;
      break;
    case gtHTT:
      type = l1t::EtSum::EtSumType::kTotalHt;
      MissingEnergy = false;
      break;
    case gtETMHF:
      type = l1t::EtSum::EtSumType::kMissingEtHF;
      MissingEnergy = true;
      break;
    case gtTowerCount:
      type = l1t::EtSum::EtSumType::kTowerCount;
      MissingEnergy = false;
      break;
    case gtMinBiasHFP0:
      type = l1t::EtSum::EtSumType::kMinBiasHFP0;
      MissingEnergy = false;
      break;
    case gtMinBiasHFM0:
      type = l1t::EtSum::EtSumType::kMinBiasHFM0;
      MissingEnergy = false;
      break;
    case gtMinBiasHFP1:
      type = l1t::EtSum::EtSumType::kMinBiasHFP1;
      MissingEnergy = false;
      break;
    case gtMinBiasHFM1:
      type = l1t::EtSum::EtSumType::kMinBiasHFM1;
      MissingEnergy = false;
      break;
    case gtAsymmetryEt:
      type = l1t::EtSum::EtSumType::kAsymEt;
      MissingEnergy = false;
      break;
    case gtAsymmetryHt:
      type = l1t::EtSum::EtSumType::kAsymHt;
      MissingEnergy = false;
      break;
    case gtAsymmetryEtHF:
      type = l1t::EtSum::EtSumType::kAsymEtHF;
      MissingEnergy = false;
      break;
    case gtAsymmetryHtHF:
      type = l1t::EtSum::EtSumType::kAsymHtHF;
      MissingEnergy = false;
      break;
    case gtCentrality0:
      centbit = 0;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality1:
      centbit = 1;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality2:
      centbit = 2;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality3:
      centbit = 3;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality4:
      centbit = 4;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality5:
      centbit = 5;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality6:
      centbit = 6;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtCentrality7:
      centbit = 7;
      type = l1t::EtSum::EtSumType::kCentrality;
      MissingEnergy = false;
      break;
    case gtZDCP:
      type = l1t::EtSum::EtSumType::kZDCP;
      MissingEnergy = false;
      break;
    case gtZDCM:
      type = l1t::EtSum::EtSumType::kZDCM;
      MissingEnergy = false;
      break;
    default:
      edm::LogError("L1TGlobal")
          << "\n  Error: "
          << "Unmatched object type from template to EtSumType, (m_gtEnergySumTemplate->objectType())[0] = "
          << (m_gtEnergySumTemplate->objectType())[0] << std::endl;
      type = l1t::EtSum::EtSumType::kTotalEt;
      break;
  }

  // get energy, phi (ETM and HTM) and overflow for the trigger object
  unsigned int candEt = 0;
  unsigned int candPhi = 0;
  bool candOverflow = false;
  for (int iEtSum = 0; iEtSum < numberObjects; ++iEtSum) {
    l1t::EtSum cand = *(candVec->at(useBx, iEtSum));
    if (cand.getType() != type)
      continue;
    candEt = cand.hwPt();
    candPhi = cand.hwPhi();
  }

  const EnergySumTemplate::ObjectParameter objPar = (*(m_gtEnergySumTemplate->objectParameter()))[iCondition];

  // check energy threshold and overflow
  // overflow evaluation:
  //     for condGEq >=
  //         candidate overflow true -> condition true
  //         candidate overflow false -> evaluate threshold
  //     for condGEq =
  //         candidate overflow true -> condition false
  //         candidate overflow false -> evaluate threshold
  //

  bool condGEqVal = m_gtEnergySumTemplate->condGEq();

  if (type == l1t::EtSum::EtSumType::kCentrality) {
    bool myres = checkBit(candEt, centbit);
    //std::cout << "CCLC:  Checking bit " << centbit << "\tResult is: " << myres << std::endl;
    if (!myres) {
      LogDebug("L1TGlobal") << "\t\t l1t::EtSum failed Centrality bit" << std::endl;
      return false;
    }
  } else if (type == l1t::EtSum::EtSumType::kZDCP || type == l1t::EtSum::EtSumType::kZDCM) {
    return false;
  } else {
    // check energy threshold
    if (!checkThreshold(objPar.etLowThreshold, objPar.etHighThreshold, candEt, condGEqVal)) {
      LogDebug("L1TGlobal") << "\t\t l1t::EtSum failed checkThreshold" << std::endl;
      return false;
    }

    if (!condGEqVal && candOverflow)
      return false;

    // for ETM and HTM check phi also
    // for overflow, the phi requirements are ignored
    if (MissingEnergy) {
      // check phi
      if (!checkRangePhi(
              candPhi, objPar.phiWindow1Lower, objPar.phiWindow1Upper, objPar.phiWindow2Lower, objPar.phiWindow2Upper)) {
        LogDebug("L1TGlobal") << "\t\t l1t::EtSum failed checkRange(phi)" << std::endl;
        return false;
      }
    }
  }

  // index is always zero, as they are global quantities (there is only one object)
  int indexObj = 0;

  objectsInComb.push_back(indexObj);
  (combinationsInCond()).push_back(objectsInComb);

  // if we get here all checks were successfull for this combination
  // set the general result for evaluateCondition to "true"

  condResult = true;
  return condResult;
}

void l1t::EnergySumCondition::print(std::ostream& myCout) const {
  m_gtEnergySumTemplate->print(myCout);
  ConditionEvaluation::print(myCout);
}
