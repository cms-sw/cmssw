// this class header
#include "L1Trigger/L1TGlobal/interface/TOPOTemplate.h"

// system include files
#include <iostream>
#include <iomanip>

TOPOTemplate::TOPOTemplate() : GlobalCondition() { m_condCategory = l1t::CondTOPO; }

TOPOTemplate::TOPOTemplate(const std::string& cName) : GlobalCondition(cName) { m_condCategory = l1t::CondTOPO; }

TOPOTemplate::TOPOTemplate(const std::string& cName, const l1t::GtConditionType& cType)  //not sure we need cType
    : GlobalCondition(cName, l1t::CondTOPO, cType) {
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.reserve(nObjects);
  }
}

// copy constructor
TOPOTemplate::TOPOTemplate(const TOPOTemplate& cp) : GlobalCondition(cp.m_condName) { copy(cp); }

// destructor
TOPOTemplate::~TOPOTemplate() {
  // empty now
}

// assign operator
TOPOTemplate& TOPOTemplate::operator=(const TOPOTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void TOPOTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

//setModelVersion - set the model version of the condition
void TOPOTemplate::setModelVersion(const std::string& modelversion) { m_modelVersion = modelversion; }

void TOPOTemplate::print(std::ostream& myCout) const {
  myCout << "\n  TOPOTemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << " [ hex ]" << std::endl;
    myCout << "    TOPOThreshold   = " << std::hex << m_objectParameter[i].minTOPOThreshold << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void TOPOTemplate::copy(const TOPOTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();  //not needed for TOPO
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();
  m_condRelativeBx = cp.condRelativeBx();

  m_modelVersion = cp.modelVersion();  // new for utm 0.12.0
  m_objectParameter = *(cp.objectParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const TOPOTemplate& result) {
  result.print(os);
  return os;
}
