/**
 * \class L1GtHfBitCountsTemplate
 *
 *
 * Description: L1 Global Trigger "HF bit counts" template.
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
#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtHfBitCountsTemplate::L1GtHfBitCountsTemplate()
        : L1GtCondition()
{

    m_condCategory = CondHfBitCounts;

}

L1GtHfBitCountsTemplate::L1GtHfBitCountsTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondHfBitCounts;

}

L1GtHfBitCountsTemplate::L1GtHfBitCountsTemplate(const std::string& cName,
        const L1GtConditionType& cType)
        : L1GtCondition(cName, CondHfBitCounts, cType)
{

    m_condCategory = CondHfBitCounts;

    // should be always 1 - they are global quantities...
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectParameter.reserve(nObjects);

        m_objectType.reserve(nObjects);
        m_objectType.assign(nObjects, HfBitCounts);
    }

}

// copy constructor
L1GtHfBitCountsTemplate::L1GtHfBitCountsTemplate(const L1GtHfBitCountsTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtHfBitCountsTemplate::~L1GtHfBitCountsTemplate()
{
    // empty now
}

// assign operator
L1GtHfBitCountsTemplate& L1GtHfBitCountsTemplate::operator= (const L1GtHfBitCountsTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void L1GtHfBitCountsTemplate::setConditionParameter(
    const std::vector<ObjectParameter>& objParameter)
{

    m_objectParameter = objParameter;

}

void L1GtHfBitCountsTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtHfBitCountsTemplate print..." << std::endl;

    L1GtCondition::print(myCout);

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {
        myCout << std::endl;
        myCout << "  Template for object " << i << std::endl;
        myCout << "    countIndex        = "
        << std::hex << m_objectParameter[i].countIndex << " [ dec ]" << std::endl;
        myCout << "    countThreshold    = "
        << std::hex << m_objectParameter[i].countThreshold << " [ hex ]" << std::endl;

    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtHfBitCountsTemplate& result)
{
    result.print(os);
    return os;

}

void L1GtHfBitCountsTemplate::copy(const L1GtHfBitCountsTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_objectParameter = *(cp.objectParameter());

}



