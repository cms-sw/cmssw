/**
 1;95;0c* \class ZdcEnergySumCondition
 *
 *
 * Description: evaluation of a CondEnergySum condition for ZDC objects.
 *
 * Implementation: This class is built following the EnergySumCondition class 
 *                 and is adapted to evaluate a condition on ZDC objects as a threshold on an energy sum value.
 *                 The object types are kZDCP and kZDCM and share the standard EtSum DataFormat.
 *
 * \author: Elisa Fontanesi and Christopher MC GINN
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/ZdcEnergySumCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ZdcEnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::ZdcEnergySumCondition::ZdcEnergySumCondition() : ConditionEvaluation() {
  //empty
}

//     from base template condition (from event setup usually)
l1t::ZdcEnergySumCondition::ZdcEnergySumCondition(const GlobalCondition* eSumTemplate, const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtZdcEnergySumTemplate(static_cast<const ZdcEnergySumTemplate*>(eSumTemplate)),
      m_uGtB(ptrGTB)

{
  // maximum number of objects received for the evaluation of the condition
  // energy sums are global quantities - one object per event

  m_condMaxNumberObjects = 1;
}

// copy constructor
void l1t::ZdcEnergySumCondition::copy(const l1t::ZdcEnergySumCondition& cp) {
  m_gtZdcEnergySumTemplate = cp.gtZdcEnergySumTemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::ZdcEnergySumCondition::ZdcEnergySumCondition(const l1t::ZdcEnergySumCondition& cp) : ConditionEvaluation() {
  copy(cp);
}

// destructor
l1t::ZdcEnergySumCondition::~ZdcEnergySumCondition() = default;

// equal operator
l1t::ZdcEnergySumCondition& l1t::ZdcEnergySumCondition::operator=(const l1t::ZdcEnergySumCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::ZdcEnergySumCondition::setGtZdcEnergySumTemplate(const ZdcEnergySumTemplate* eSumTempl) {
  m_gtZdcEnergySumTemplate = eSumTempl;
}

// set the pointer to uGT GlobalBoard
void l1t::ZdcEnergySumCondition::setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

// try all object permutations and check spatial correlations, if required
const bool l1t::ZdcEnergySumCondition::evaluateCondition(const int bxEval) const {
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

  const BXVector<const l1t::EtSum*>* candVecZdc = m_uGtB->getCandL1ZdcEtSum();

  // Look at objects in bx = bx + relativeBx
  int useBx = bxEval + m_gtZdcEnergySumTemplate->condRelativeBx();

  // Fail condition if attempting to get Bx outside of range
  if ((useBx < candVecZdc->getFirstBX()) || (useBx > candVecZdc->getLastBX())) {
    return false;
  }

  // If no candidates, no use looking any further
  int numberObjectsZdc = candVecZdc->size(useBx);

  if (numberObjectsZdc < 1) {
    return false;
  }

  const ZdcEnergySumTemplate::ObjectParameter objPar = (*(m_gtZdcEnergySumTemplate->objectParameter()))[iCondition];
  bool condGEqVal = m_gtZdcEnergySumTemplate->condGEq();

  l1t::EtSum candZdcPlus;
  l1t::EtSum candZdcMinus;
  unsigned int candZDCPEsum = 0;
  unsigned int candZDCMEsum = 0;
  bool myres = false;

  for (int iEtSum = 0; iEtSum < numberObjectsZdc; ++iEtSum) {
    l1t::EtSum candZdc = *(candVecZdc->at(useBx + 2, iEtSum));
    // NOTE: usage of Bx 2 has to be fixed. At the moment it corresponds to 0

    if (l1t::EtSum::EtSumType::kZDCP == candZdc.getType())
      candZdcPlus = *(candVecZdc->at(useBx + 2, iEtSum));
    else if (l1t::EtSum::EtSumType::kZDCM == candZdc.getType())
      candZdcMinus = *(candVecZdc->at(useBx + 2, iEtSum));
    LogDebug("L1TGlobal") << "CANDZdc: " << candZdc.hwPt() << ", " << useBx << ", " << candZdc.getType() << std::endl;

    if (candZdc.getType() == l1t::EtSum::EtSumType::kZDCP) {
      candZDCPEsum = candZdcPlus.hwPt();
      myres = checkThreshold(objPar.etLowThreshold, objPar.etHighThreshold, candZDCPEsum, condGEqVal);
    } else if (candZdc.getType() == l1t::EtSum::EtSumType::kZDCM) {
      candZDCMEsum = candZdcMinus.hwPt();
      myres = checkThreshold(objPar.etLowThreshold, objPar.etHighThreshold, candZDCMEsum, condGEqVal);
    }

    else {
      LogDebug("L1TGlobal") << "\t\t l1t::EtSum failed ZDC checkThreshold" << std::endl;
      return false;
    }

    LogDebug("L1TGlobal")
        << "----------------------------------------------> ZDC EtSumType object from ZdcEnergySumTemplate"
        << "\n objPar.etLowThreshold = " << objPar.etLowThreshold
        << "\n objPar.etHighThreshold = " << objPar.etHighThreshold << "\n candZDCPEsum = " << candZDCPEsum
        << "\n candZDCMEsum = " << candZDCMEsum << "\n condGEqVal = " << condGEqVal << "\n myres = " << myres
        << std::endl;
  }

  if (!condGEqVal)
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

void l1t::ZdcEnergySumCondition::print(std::ostream& myCout) const {
  m_gtZdcEnergySumTemplate->print(myCout);
  ConditionEvaluation::print(myCout);
}
