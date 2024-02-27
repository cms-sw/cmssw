#include "L1Trigger/L1TGlobal/interface/CICADACondition.h"

#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

#include "L1Trigger/L1TGlobal/interface/CICADATemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

l1t::CICADACondition::CICADACondition() : ConditionEvaluation() {}

l1t::CICADACondition::CICADACondition(const GlobalCondition* cicadaTemplate, const GlobalBoard* ptrGTB)
    : ConditionEvaluation(), m_gtCICADATemplate(static_cast<const CICADATemplate*>(cicadaTemplate)), m_uGtB(ptrGTB) {
  m_condMaxNumberObjects = 1;  //necessary?
}

void l1t::CICADACondition::copy(const l1t::CICADACondition& cp) {
  m_gtCICADATemplate = cp.gtCICADATemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::CICADACondition::CICADACondition(const l1t::CICADACondition& cp) : ConditionEvaluation() { copy(cp); }

l1t::CICADACondition& l1t::CICADACondition::operator=(const l1t::CICADACondition& cp) {
  copy(cp);
  return *this;
}

const bool l1t::CICADACondition::evaluateCondition(const int bxEval) const {
  bool condResult = false;
  const float cicadaScore = m_uGtB->getCICADAScore();  //needs to be implemented

  // This gets rid of a GT emulator convention "iCondition".
  // This usually indexes the next line, which is somewhat concerning
  // AXOL1TL operates this way, but it should be checked
  const CICADATemplate::ObjectParameter objPar = (*(m_gtCICADATemplate->objectParameter()))[0];

  bool condGEqVal = m_gtCICADATemplate->condGEq();
  bool passCondition = false;

  passCondition = checkCut(objPar.minCICADAThreshold, cicadaScore, condGEqVal);

  condResult |= passCondition;

  return condResult;
}

void l1t::CICADACondition::print(std::ostream& myCout) const {
  myCout << "CICADA Condition Print: " << std::endl;
  m_gtCICADATemplate->print(myCout);
  ConditionEvaluation::print(myCout);
}
