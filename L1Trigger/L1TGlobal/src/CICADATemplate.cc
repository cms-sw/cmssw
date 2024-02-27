#include "L1Trigger/L1TGlobal/interface/CICADATemplate.h"

#include <iostream>
#include <iomanip>

//TODO: Check the actual conditions name: "CondCICADA"
CICADATemplate::CICADATemplate() : GlobalCondition() { m_condCategory = l1t::CondCICADA; }

CICADATemplate::CICADATemplate(const std::string& cName) : GlobalCondition(cName) { m_condCategory = l1t::CondCICADA; }

CICADATemplate::CICADATemplate(const std::string& cName, const l1t::GtConditionType& cType)
    : GlobalCondition(cName, l1t::CondCICADA, cType) {
  m_condCategory = l1t::CondCICADA;

  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.reserve(nObjects);
  }
}

CICADATemplate::CICADATemplate(const CICADATemplate& cp) : GlobalCondition(cp.m_condName) { copy(cp); }

CICADATemplate& CICADATemplate::operator=(const CICADATemplate& cp) {
  copy(cp);
  return *this;
}

void CICADATemplate::print(std::ostream& myCout) const {
  myCout << "\n CICADATemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; ++i) {
    myCout << std::endl;
    myCout << " Template for object " << i << " [ hex ]" << std::endl;
    myCout << " CICADAThreshold      = " << std::hex << m_objectParameter[i].minCICADAThreshold << std::endl;
  }

  myCout << std::dec << std::endl;
}

void CICADATemplate::copy(const CICADATemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();
  m_condRelativeBx = cp.condRelativeBx();

  m_objectParameter = *(cp.objectParameter());
}

std::ostream& operator<<(std::ostream& os, const CICADATemplate& result) {
  result.print(os);
  return os;
}
