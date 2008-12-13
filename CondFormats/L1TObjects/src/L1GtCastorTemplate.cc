/**
 * \class L1GtCastorTemplate
 *
 *
 * Description: L1 Global Trigger CASTOR template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. CASTOR conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. CASTOR conditions can be used
 *    in physics algorithms in combination with muon, calorimeter, energy sum
 *    and jet-counts conditions.
 *    It has zero objects.
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtCastorTemplate::L1GtCastorTemplate()
        : L1GtCondition()
{

    m_condCategory = CondCastor;
    m_condType = TypeCastor;

}

L1GtCastorTemplate::L1GtCastorTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondCastor;
    m_condType = TypeCastor;

}

L1GtCastorTemplate::L1GtCastorTemplate(const std::string& cName, const L1GtConditionType& cType)
        : L1GtCondition(cName, CondEnergySum, cType)
{

    m_condCategory = CondCastor;
    m_condType = TypeCastor;

    // actually no objects are sent by CASTOR, only the result of the condition
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

}

// copy constructor
L1GtCastorTemplate::L1GtCastorTemplate(const L1GtCastorTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtCastorTemplate::~L1GtCastorTemplate()
{
    // empty now
}

// assign operator
L1GtCastorTemplate& L1GtCastorTemplate::operator= (const L1GtCastorTemplate& cp)
{

    copy(cp);
    return *this;
}


void L1GtCastorTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtCastorTemplate print..." << std::endl;

    L1GtCondition::print(myCout);


    // reset to decimal output
    myCout << std::dec << std::endl;
}

void L1GtCastorTemplate::copy(const L1GtCastorTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtCastorTemplate& result)
{
    result.print(os);
    return os;

}


