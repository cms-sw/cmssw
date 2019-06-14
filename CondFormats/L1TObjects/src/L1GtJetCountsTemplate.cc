/**
 * \class L1GtJetCountsTemplate
 *
 *
 * Description: L1 Global Trigger "jet counts" template.
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
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtJetCountsTemplate::L1GtJetCountsTemplate() : L1GtCondition() { m_condCategory = CondJetCounts; }

L1GtJetCountsTemplate::L1GtJetCountsTemplate(const std::string& cName) : L1GtCondition(cName) {
  m_condCategory = CondJetCounts;
}

L1GtJetCountsTemplate::L1GtJetCountsTemplate(const std::string& cName, const L1GtConditionType& cType)
    : L1GtCondition(cName, CondJetCounts, cType) {
  m_condCategory = CondJetCounts;

  // should be always 1 - they are global quantities...
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectParameter.reserve(nObjects);

    m_objectType.reserve(nObjects);
    m_objectType.assign(nObjects, JetCounts);
  }
}

// copy constructor
L1GtJetCountsTemplate::L1GtJetCountsTemplate(const L1GtJetCountsTemplate& cp) : L1GtCondition(cp.m_condName) {
  copy(cp);
}

// destructor
L1GtJetCountsTemplate::~L1GtJetCountsTemplate() {
  // empty now
}

// assign operator
L1GtJetCountsTemplate& L1GtJetCountsTemplate::operator=(const L1GtJetCountsTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void L1GtJetCountsTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

void L1GtJetCountsTemplate::print(std::ostream& myCout) const {
  myCout << "\n  L1GtJetCountsTemplate print..." << std::endl;

  L1GtCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << std::endl;
    myCout << "    countIndex        = " << std::hex << m_objectParameter[i].countIndex << " [ dec ]" << std::endl;
    myCout << "    countThreshold    = " << std::hex << m_objectParameter[i].countThreshold << " [ hex ]" << std::endl;
    myCout << "    countOverflow     = " << std::hex << m_objectParameter[0].countOverflow << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void L1GtJetCountsTemplate::copy(const L1GtJetCountsTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();

  m_objectParameter = *(cp.objectParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtJetCountsTemplate& result) {
  result.print(os);
  return os;
}
