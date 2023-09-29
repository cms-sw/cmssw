/**
\class EnergySumZdcCondition
 *
 *
 * Description: evaluation of a CondEnergySum condition for ZDC objects.
 *
 * Implementation: This class is built following the EnergySumCondition class 
 *                 and is adapted to evaluate a condition on ZDC objects as a threshold on an energy sum value.
 *                 The object types are kZDCP and kZDCM and share the standard EtSum DataFormat.
 *
 * \author: Elisa Fontanesi and Christopher Mc Ginn
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::EnergySumZdcCondition::EnergySumZdcCondition() : ConditionEvaluation() {
  //empty
}

//     from base template condition (from event setup usually)
l1t::EnergySumZdcCondition::EnergySumZdcCondition(const GlobalCondition* eSumTemplate, const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtEnergySumZdcTemplate(static_cast<const EnergySumZdcTemplate*>(eSumTemplate)),
      m_uGtB(ptrGTB)

{
  // maximum number of objects received for the evaluation of the condition
  // energy sums are global quantities - one object per event

  m_condMaxNumberObjects = 1;
}

// copy constructor
void l1t::EnergySumZdcCondition::copy(const l1t::EnergySumZdcCondition& cp) {
  m_gtEnergySumZdcTemplate = cp.gtEnergySumZdcTemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::EnergySumZdcCondition::EnergySumZdcCondition(const l1t::EnergySumZdcCondition& cp) : ConditionEvaluation() {
  copy(cp);
}

// destructor
l1t::EnergySumZdcCondition::~EnergySumZdcCondition() = default;

// equal operator
l1t::EnergySumZdcCondition& l1t::EnergySumZdcCondition::operator=(const l1t::EnergySumZdcCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::EnergySumZdcCondition::setGtEnergySumZdcTemplate(const EnergySumZdcTemplate* eSumTempl) {
  m_gtEnergySumZdcTemplate = eSumTempl;
}

// set the pointer to uGT GlobalBoard
void l1t::EnergySumZdcCondition::setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

// try all object permutations and check spatial correlations, if required
const bool l1t::EnergySumZdcCondition::evaluateCondition(const int bxEval) const {
  // number of trigger objects in the condition: there is only one object
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

  const BXVector<const l1t::EtSum*>* candVecZdc = m_uGtB->getCandL1EtSumZdc();

  // Look at objects in bx = bx + relativeBx
  int useBx = bxEval + m_gtEnergySumZdcTemplate->condRelativeBx();

  // Fail condition if attempting to get Bx outside of range
  if ((useBx < candVecZdc->getFirstBX()) || (useBx > candVecZdc->getLastBX())) {
    return false;
  }

  // If no candidates, no use looking any further
  int numberObjectsZdc = candVecZdc->size(useBx);

  if (numberObjectsZdc < 1) {
    return false;
  }

  const EnergySumZdcTemplate::ObjectParameter objPar = (*(m_gtEnergySumZdcTemplate->objectParameter()))[iCondition];
  l1t::EtSum::EtSumType type;
  switch ((m_gtEnergySumZdcTemplate->objectType())[0]) {
    case gtZDCP:
      type = l1t::EtSum::EtSumType::kZDCP;
      break;
    case gtZDCM:
      type = l1t::EtSum::EtSumType::kZDCM;
      break;
    default:
      edm::LogError("L1TGlobal")
          << "\n  Error: "
          << "Unmatched object type from template to EtSumZdcType, (m_gtEnergySumZdcTemplate->objectType())[0] = "
          << (m_gtEnergySumZdcTemplate->objectType())[0] << std::endl;
      type = l1t::EtSum::EtSumType::kZDCP;
      break;
  }

  // Definition in CondFormats/L1TObjects/interface/L1GtCondition.h:
  // condGEqVal indicates the operator used for the condition (>=, =): true for >=
  bool condGEqVal = m_gtEnergySumZdcTemplate->condGEq();

  l1t::EtSum candZdcPlus;
  l1t::EtSum candZdcMinus;
  unsigned int candZDCPEsum = 0;
  unsigned int candZDCMEsum = 0;
  bool myres = false;

  for (int iEtSum = 0; iEtSum < numberObjectsZdc; ++iEtSum) {
    l1t::EtSum candZdc = *(candVecZdc->at(useBx, iEtSum));

    if (candZdc.getType() != type)
      continue;

    if (candZdc.getType() == l1t::EtSum::EtSumType::kZDCP) {
      candZdcPlus = *(candVecZdc->at(useBx, iEtSum));
      candZDCPEsum = candZdcPlus.hwPt();
      myres = checkThreshold(objPar.etLowThreshold, objPar.etHighThreshold, candZDCPEsum, condGEqVal);
    } else if (candZdc.getType() == l1t::EtSum::EtSumType::kZDCM) {
      candZdcMinus = *(candVecZdc->at(useBx, iEtSum));
      candZDCMEsum = candZdcMinus.hwPt();
      myres = checkThreshold(objPar.etLowThreshold, objPar.etHighThreshold, candZDCMEsum, condGEqVal);
    } else {
      LogDebug("L1TGlobal") << "\t\t l1t::EtSum failed ZDC checkThreshold" << std::endl;
      return false;
    }

    LogDebug("L1TGlobal") << "CANDZdc: " << candZdc.hwPt() << ", " << useBx << ", " << candZdc.getType();

    LogDebug("L1TGlobal")
        << "----------------------------------------------> ZDC EtSumType object from EnergySumZdcTemplate"
        << "\n objPar.etLowThreshold = " << objPar.etLowThreshold
        << "\n objPar.etHighThreshold = " << objPar.etHighThreshold << "\n candZDCPEsum = " << candZDCPEsum
        << "\n candZDCMEsum = " << candZDCMEsum << "\n condGEqVal = " << condGEqVal << "\n myres = " << myres
        << std::endl;
  }

  if (not myres)
    return false;

  // index is always zero, as they are global quantities (there is only one object)
  int indexObj = 0;

  objectsInComb.push_back(indexObj);
  (combinationsInCond()).push_back(objectsInComb);

  // if we get here all checks were successful for this combination
  // set the general result for evaluateCondition to "true"
  condResult = true;
  return condResult;
}

void l1t::EnergySumZdcCondition::print(std::ostream& myCout) const {
  m_gtEnergySumZdcTemplate->print(myCout);
  ConditionEvaluation::print(myCout);
}
