/**
 * \class L1GtExternalTemplate
 *
 *
 * Description: L1 Global Trigger external template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. External conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. External conditions can be used
 *    in physics algorithms in combination with other defined conditions,
 *    see L1GtFwd.
 *
 *    It has zero objects associated.
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtExternalTemplate::L1GtExternalTemplate()
        : L1GtCondition()
{

    m_condCategory = CondExternal;
    m_condType = TypeExternal;

}

L1GtExternalTemplate::L1GtExternalTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondExternal;
    m_condType = TypeExternal;

}

L1GtExternalTemplate::L1GtExternalTemplate(const std::string& cName, const L1GtConditionType& cType)
        : L1GtCondition(cName, CondEnergySum, cType)
{

    m_condCategory = CondExternal;
    m_condType = TypeExternal;

    // actually no objects are sent by External, only the result of the condition
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

}

// copy constructor
L1GtExternalTemplate::L1GtExternalTemplate(const L1GtExternalTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtExternalTemplate::~L1GtExternalTemplate()
{
    // empty now
}

// assign operator
L1GtExternalTemplate& L1GtExternalTemplate::operator= (const L1GtExternalTemplate& cp)
{

    copy(cp);
    return *this;
}


void L1GtExternalTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtExternalTemplate print..." << std::endl;

    L1GtCondition::print(myCout);


    // reset to decimal output
    myCout << std::dec << std::endl;
}

void L1GtExternalTemplate::copy(const L1GtExternalTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtExternalTemplate& result)
{
    result.print(os);
    return os;

}


