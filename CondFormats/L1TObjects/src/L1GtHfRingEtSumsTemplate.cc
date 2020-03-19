/**
 * \class L1GtHfRingEtSumsTemplate
 *
 *
 * Description: L1 Global Trigger "HF Ring ET sums" template.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtHfRingEtSumsTemplate::L1GtHfRingEtSumsTemplate() : L1GtCondition() { m_condCategory = CondHfRingEtSums; }

L1GtHfRingEtSumsTemplate::L1GtHfRingEtSumsTemplate(const std::string& cName) : L1GtCondition(cName) {
  m_condCategory = CondHfRingEtSums;
}

L1GtHfRingEtSumsTemplate::L1GtHfRingEtSumsTemplate(const std::string& cName, const L1GtConditionType& cType)
    : L1GtCondition(cName, CondHfRingEtSums, cType) {
  m_condCategory = CondHfRingEtSums;

  // should be always 1 - they are global quantities...
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectParameter.reserve(nObjects);

    m_objectType.reserve(nObjects);
    m_objectType.assign(nObjects, HfRingEtSums);
  }
}

// copy constructor
L1GtHfRingEtSumsTemplate::L1GtHfRingEtSumsTemplate(const L1GtHfRingEtSumsTemplate& cp) : L1GtCondition(cp.m_condName) {
  copy(cp);
}

// destructor
L1GtHfRingEtSumsTemplate::~L1GtHfRingEtSumsTemplate() {
  // empty now
}

// assign operator
L1GtHfRingEtSumsTemplate& L1GtHfRingEtSumsTemplate::operator=(const L1GtHfRingEtSumsTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void L1GtHfRingEtSumsTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

void L1GtHfRingEtSumsTemplate::print(std::ostream& myCout) const {
  myCout << "\n  L1GtHfRingEtSumsTemplate print..." << std::endl;

  L1GtCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << std::endl;
    myCout << "    etSumIndex        = " << std::hex << m_objectParameter[i].etSumIndex << " [ dec ]" << std::endl;
    myCout << "    etSumThreshold    = " << std::hex << m_objectParameter[i].etSumThreshold << " [ hex ]" << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtHfRingEtSumsTemplate& result) {
  result.print(os);
  return os;
}

void L1GtHfRingEtSumsTemplate::copy(const L1GtHfRingEtSumsTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();

  m_objectParameter = *(cp.objectParameter());
}
