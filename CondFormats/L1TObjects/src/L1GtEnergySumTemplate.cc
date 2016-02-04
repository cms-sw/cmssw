/**
 * \class L1GtEnergySumTemplate
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
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtEnergySumTemplate::L1GtEnergySumTemplate()
        : L1GtCondition()
{

    m_condCategory = CondEnergySum;

}

L1GtEnergySumTemplate::L1GtEnergySumTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondEnergySum;

}

L1GtEnergySumTemplate::L1GtEnergySumTemplate(const std::string& cName, const L1GtConditionType& cType)
        : L1GtCondition(cName, CondEnergySum, cType)
{

    m_condCategory = CondEnergySum;

    // should be always 1 - they are global quantities...
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectParameter.reserve(nObjects);
        m_objectType.reserve(nObjects);
    }

}

// copy constructor
L1GtEnergySumTemplate::L1GtEnergySumTemplate(const L1GtEnergySumTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtEnergySumTemplate::~L1GtEnergySumTemplate()
{
    // empty now
}

// assign operator
L1GtEnergySumTemplate& L1GtEnergySumTemplate::operator= (const L1GtEnergySumTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void L1GtEnergySumTemplate::setConditionParameter(
    const std::vector<ObjectParameter>& objParameter)
{

    m_objectParameter = objParameter;

}

void L1GtEnergySumTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtEnergySumTemplate print..." << std::endl;

    L1GtCondition::print(myCout);

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {
        myCout << std::endl;
        myCout << "  Template for object " << i << " [ hex ]" << std::endl;
        myCout << "    etThreshold       = "
        << std::hex << m_objectParameter[i].etThreshold << std::endl;
        myCout << "    energyOverflow    = "
        <<  std::hex << m_objectParameter[0].energyOverflow << std::endl;

        if (m_condType == TypeETM) {
            myCout << "    phi               = "
            << std::hex << m_objectParameter[i].phiRange1Word
            << std::hex << m_objectParameter[i].phiRange0Word
            << std::endl;
        } else if (m_condType == TypeHTM) {
            myCout << "    phi               = "
            << std::hex << m_objectParameter[i].phiRange0Word
            << std::endl;
        }

    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void L1GtEnergySumTemplate::copy(const L1GtEnergySumTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_objectParameter = *(cp.objectParameter());

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtEnergySumTemplate& result)
{
    result.print(os);
    return os;

}


