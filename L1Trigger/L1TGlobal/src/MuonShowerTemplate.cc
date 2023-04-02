// this class header
#include "L1Trigger/L1TGlobal/interface/MuonShowerTemplate.h"

// system include files
#include <iostream>
#include <iomanip>

MuonShowerTemplate::MuonShowerTemplate() : GlobalCondition() { m_condCategory = l1t::CondMuonShower; }

MuonShowerTemplate::MuonShowerTemplate(const std::string& cName) : GlobalCondition(cName) {
  m_condCategory = l1t::CondMuonShower;
}

MuonShowerTemplate::MuonShowerTemplate(const std::string& cName, const l1t::GtConditionType& cType)
    : GlobalCondition(cName, l1t::CondMuonShower, cType) {
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectParameter.reserve(nObjects);

    m_objectType.reserve(nObjects);
    m_objectType.assign(nObjects, l1t::gtMuShower);
  }
}

// copy constructor
MuonShowerTemplate::MuonShowerTemplate(const MuonShowerTemplate& cp) : GlobalCondition(cp.m_condName) { copy(cp); }

// destructor
MuonShowerTemplate::~MuonShowerTemplate() {
  // empty now
}

// assign operator
MuonShowerTemplate& MuonShowerTemplate::operator=(const MuonShowerTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void MuonShowerTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

void MuonShowerTemplate::print(std::ostream& myCout) const {
  myCout << "\n  MuonShowerTemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << " [ hex ]" << std::endl;
    myCout << "    MuonShower0   = " << std::hex << m_objectParameter[i].MuonShower0 << std::endl;
    myCout << "    MuonShower1   = " << std::hex << m_objectParameter[i].MuonShower1 << std::endl;
    myCout << "    MuonShower2   = " << std::hex << m_objectParameter[i].MuonShower2 << std::endl;
    myCout << "    MuonShowerOutOfTime0   = " << std::hex << m_objectParameter[i].MuonShowerOutOfTime0 << std::endl;
    myCout << "    MuonShowerOutOfTime1   = " << std::hex << m_objectParameter[i].MuonShowerOutOfTime1 << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void MuonShowerTemplate::copy(const MuonShowerTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();
  m_condRelativeBx = cp.condRelativeBx();

  m_objectParameter = *(cp.objectParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const MuonShowerTemplate& result) {
  result.print(os);
  return os;
}
