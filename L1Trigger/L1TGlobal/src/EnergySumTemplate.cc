/**
 * \class EnergySumTemplate
 *
 *
 * Description: L1 Global Trigger energy-sum template.
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
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class


// forward declarations

// constructors
EnergySumTemplate::EnergySumTemplate()
        : GlobalCondition()
{

    m_condCategory = l1t::CondEnergySum;

}

EnergySumTemplate::EnergySumTemplate(const std::string& cName)
        : GlobalCondition(cName)
{

    m_condCategory = l1t::CondEnergySum;

}

EnergySumTemplate::EnergySumTemplate(const std::string& cName, const l1t::GtConditionType& cType)
        : GlobalCondition(cName, l1t::CondEnergySum, cType)
{

    m_condCategory = l1t::CondEnergySum;

    // should be always 1 - they are global quantities...
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectParameter.reserve(nObjects);
        m_objectType.reserve(nObjects);
    }

}

// copy constructor
EnergySumTemplate::EnergySumTemplate(const EnergySumTemplate& cp)
        : GlobalCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
EnergySumTemplate::~EnergySumTemplate()
{
    // empty now
}

// assign operator
EnergySumTemplate& EnergySumTemplate::operator= (const EnergySumTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void EnergySumTemplate::setConditionParameter(
    const std::vector<ObjectParameter>& objParameter)
{

    m_objectParameter = objParameter;

}

void EnergySumTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  EnergySumTemplate print..." << std::endl;

    GlobalCondition::print(myCout);

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {
        myCout << std::endl;
        myCout << "  Template for object " << i << " [ hex ]" << std::endl;
        myCout << "    etThreshold       = "
        << std::hex << m_objectParameter[i].etLowThreshold << " - " << m_objectParameter[i].etHighThreshold << std::endl;
        myCout << "    energyOverflow    = "
        <<  std::hex << m_objectParameter[0].energyOverflow << std::endl;

        if (m_condType == l1t::TypeETM) {
            myCout << "    phi               = "
            << std::hex << m_objectParameter[i].phiRange1Word
            << std::hex << m_objectParameter[i].phiRange0Word
            << std::endl;
        } else if (m_condType == l1t::TypeHTM) {
            myCout << "    phi               = "
            << std::hex << m_objectParameter[i].phiRange0Word
            << std::endl;
        } else if (m_condType == l1t::TypeETM2) {
            myCout << "    phi               = "
            << std::hex << m_objectParameter[i].phiRange0Word
            << std::endl;
        }

    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void EnergySumTemplate::copy(const EnergySumTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();
    m_condRelativeBx = cp.condRelativeBx();

    m_objectParameter = *(cp.objectParameter());

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const EnergySumTemplate& result)
{
    result.print(os);
    return os;

}


