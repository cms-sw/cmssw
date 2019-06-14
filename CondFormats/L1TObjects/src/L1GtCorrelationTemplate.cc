/**
 * \class L1GtCorrelationTemplate
 *
 *
 * Description: L1 Global Trigger correlation template.
 * Includes spatial correlation for two objects of different type.
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
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files
//   base class

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
//   default

L1GtCorrelationTemplate::L1GtCorrelationTemplate() : L1GtCondition() {
  m_condCategory = CondCorrelation;
  m_condType = Type2cor;
  m_condChipNr = -1;

  // there are in fact two objects
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.reserve(nObjects);
  }

  m_cond0Category = CondNull;
  m_cond1Category = CondNull;
  m_cond0Index = -1;
  m_cond1Index = -1;
}

//   from condition name
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const std::string& cName) : L1GtCondition(cName) {
  m_condCategory = CondCorrelation;
  m_condType = Type2cor;
  m_condChipNr = -1;

  // there are in fact two objects
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.reserve(nObjects);
  }

  m_cond0Category = CondNull;
  m_cond1Category = CondNull;
  m_cond0Index = -1;
  m_cond1Index = -1;
}

//   from condition name, the category of first sub-condition, the category of the
//   second sub-condition, the index of first sub-condition in the cor* vector,
//   the index of second sub-condition in the cor* vector
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const std::string& cName,
                                                 const L1GtConditionCategory& cond0Cat,
                                                 const L1GtConditionCategory& cond1Cat,
                                                 const int cond0Index,
                                                 const int cond1index)
    : L1GtCondition(cName),
      m_cond0Category(cond0Cat),
      m_cond1Category(cond1Cat),
      m_cond0Index(cond0Index),
      m_cond1Index(cond1index)

{
  m_condCategory = CondCorrelation;
  m_condType = Type2cor;
  m_condChipNr = -1;

  // there are in fact two objects
  int nObjects = nrObjects();

  if (nObjects > 0) {
    m_objectType.resize(nObjects);
  }
}

// copy constructor
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const L1GtCorrelationTemplate& cp) : L1GtCondition(cp.m_condName) {
  copy(cp);
}

// destructor
L1GtCorrelationTemplate::~L1GtCorrelationTemplate() {
  // empty now
}

// assign operator
L1GtCorrelationTemplate& L1GtCorrelationTemplate::operator=(const L1GtCorrelationTemplate& cp) {
  copy(cp);
  return *this;
}

// set the category of the two sub-conditions
void L1GtCorrelationTemplate::setCond0Category(const L1GtConditionCategory& condCateg) { m_cond0Category = condCateg; }

void L1GtCorrelationTemplate::setCond1Category(const L1GtConditionCategory& condCateg) { m_cond1Category = condCateg; }

// set the index of the two sub-conditions in the cor* vector from menu
void L1GtCorrelationTemplate::setCond0Index(const int& condIndex) { m_cond0Index = condIndex; }

void L1GtCorrelationTemplate::setCond1Index(const int& condIndex) { m_cond1Index = condIndex; }

// set the correlation parameters of the condition
void L1GtCorrelationTemplate::setCorrelationParameter(const CorrelationParameter& corrParameter) {
  m_correlationParameter = corrParameter;
}

void L1GtCorrelationTemplate::print(std::ostream& myCout) const {
  myCout << "\n  L1GtCorrelationTemplate print..." << std::endl;

  L1GtCondition::print(myCout);

  myCout << "\n  First sub-condition category:  " << m_cond0Category << std::endl;
  myCout << "  Second sub-condition category: " << m_cond1Category << std::endl;

  myCout << "\n  First sub-condition index:  " << m_cond0Index << std::endl;
  myCout << "  Second sub-condition index: " << m_cond1Index << std::endl;

  myCout << "\n  Correlation parameters "
         << "[ hex ]" << std::endl;

  myCout << "    deltaEtaRange      = " << std::hex << m_correlationParameter.deltaEtaRange << std::endl;
  myCout << "    deltaPhiRange      = " << std::hex << m_correlationParameter.deltaPhiRange << std::endl;
  myCout << "    deltaPhiMaxbits    = " << std::hex << m_correlationParameter.deltaPhiMaxbits << std::endl;

  // reset to decimal output
  myCout << std::dec << std::endl;
}

void L1GtCorrelationTemplate::copy(const L1GtCorrelationTemplate& cp) {
  m_condName = cp.condName();
  m_condCategory = cp.condCategory();
  m_condType = cp.condType();
  m_objectType = cp.objectType();
  m_condGEq = cp.condGEq();
  m_condChipNr = cp.condChipNr();

  m_cond0Category = cp.cond0Category();
  m_cond1Category = cp.cond1Category();
  m_cond0Index = cp.cond0Index();
  m_cond1Index = cp.cond1Index();

  m_correlationParameter = *(cp.correlationParameter());
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtCorrelationTemplate& result) {
  result.print(os);
  return os;
}
