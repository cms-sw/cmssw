/**
 * \class EnergySumZdcTemplate
 *
 *
 * Description: L1 Global Trigger energy-sum template for EtSums from ZDC.
 *
 * Implementation:
 *    Effective clone of EnergySumTemplate for use w/ ZDC
 *
 * \author: Elisa Fontanesi, Boston University, and Christopher Mc Ginn, MIT
 *   cloned from work of Vasile Mihai Ghete - HEPHY Vienna
 *
 * 2023.08.31
 * Version 1
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

// forward declarations

// constructors
EnergySumZdcTemplate::EnergySumZdcTemplate() : GlobalCondition() { m_condCategory = l1t::CondEnergySumZdc; }

EnergySumZdcTemplate::EnergySumZdcTemplate(const std::string& cName) : GlobalCondition(cName) {
  m_condCategory = l1t::CondEnergySumZdc;
}

EnergySumZdcTemplate::EnergySumZdcTemplate(const std::string& cName, const l1t::GtConditionType& cType)
    : GlobalCondition(cName, l1t::CondEnergySumZdc, cType) {
  m_condCategory = l1t::CondEnergySumZdc;

  // should be always 1 - they are global quantities...
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectParameter.reserve(nObjects);
    m_objectType.reserve(nObjects);
  }
}

// copy constructor
EnergySumZdcTemplate::EnergySumZdcTemplate(const EnergySumZdcTemplate& cp) : GlobalCondition(cp.m_condName) {
  copy(cp);
}

// destructor
EnergySumZdcTemplate::~EnergySumZdcTemplate() = default;

// assign operator
EnergySumZdcTemplate& EnergySumZdcTemplate::operator=(const EnergySumZdcTemplate& cp) {
  copy(cp);
  return *this;
}

// setConditionParameter - set the parameters of the condition
void EnergySumZdcTemplate::setConditionParameter(const std::vector<ObjectParameter>& objParameter) {
  m_objectParameter = objParameter;
}

void EnergySumZdcTemplate::print(std::ostream& myCout) const {
  myCout << "\n  EnergySumZdcTemplate print..." << std::endl;

  GlobalCondition::print(myCout);

  int nObjects = nrObjects();

  for (int i = 0; i < nObjects; i++) {
    myCout << std::endl;
    myCout << "  Template for object " << i << " [ hex ]" << std::endl;
    myCout << "    etThreshold       = " << std::hex << m_objectParameter[i].etLowThreshold << " - "
           << m_objectParameter[i].etHighThreshold << std::endl;
  }

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void EnergySumZdcTemplate::copy(const EnergySumZdcTemplate& cp) {
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
std::ostream& operator<<(std::ostream& os, const EnergySumZdcTemplate& result) {
  result.print(os);
  return os;
}
