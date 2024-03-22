// this class header
#include "L1Trigger/L1TGlobal/interface/AXOL1TLTemplate.h"

// system include files
#include <iostream>
#include <iomanip>

AXOL1TLTemplate::AXOL1TLTemplate() : GlobalCondition() { m_condCategory = l1t::CondAXOL1TL; }

AXOL1TLTemplate::AXOL1TLTemplate(const std::string& cName) : GlobalCondition(cName) {
  m_condCategory = l1t::CondAXOL1TL;
}

AXOL1TLTemplate::AXOL1TLTemplate(const std::string& cName, const l1t::GtConditionType& cType)  //not sure we need cType
    : GlobalCondition(cName, l1t::CondAXOL1TL, cType) {
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.reserve(nObjects);
  }
}

// copy constructor
AXOL1TLTemplate::AXOL1TLTemplate(const AXOL1TLTemplate& cp) : GlobalCondition(cp.m_condName) { copy(cp); }

// destructor
AXOL1TLTemplate::~AXOL1TLTemplate() {
  // empty now
}

// assign operator
AXOL1TLTemplate& AXOL1TLTemplate::operator=(const AXOL1TLTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void AXOL1TLTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

void AXOL1TLTemplate::print(std::ostream& myCout) const {
  myCout << "\n  AXOL1TLTemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << " [ hex ]" << std::endl;
    myCout << "    AXOL1TLThreshold   = " << std::hex << m_objectParameter[i].minAXOL1TLThreshold << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void AXOL1TLTemplate::copy(const AXOL1TLTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();  //not needed for AXOL1TL
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();
  m_condRelativeBx = cp.condRelativeBx();

  m_objectParameter = *(cp.objectParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const AXOL1TLTemplate& result) {
  result.print(os);
  return os;
}
