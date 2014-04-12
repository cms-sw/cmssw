/**
 * \class L1GtBptxTemplate
 *
 *
 * Description: L1 Global Trigger BPTX template.
 *
 * Implementation:
 *    Instantiated L1GtCondition. BPTX conditions sends a logical result only.
 *    No changes are possible at the L1 GT level. BPTX conditions can be used
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
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtBptxTemplate::L1GtBptxTemplate()
        : L1GtCondition()
{

    m_condCategory = CondBptx;
    m_condType = TypeBptx;

}

L1GtBptxTemplate::L1GtBptxTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondBptx;
    m_condType = TypeBptx;

}

L1GtBptxTemplate::L1GtBptxTemplate(const std::string& cName, const L1GtConditionType& cType)
        : L1GtCondition(cName, CondEnergySum, cType)
{

    m_condCategory = CondBptx;
    m_condType = TypeBptx;

    // actually no objects are sent by BPTX, only the result of the condition
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

}

// copy constructor
L1GtBptxTemplate::L1GtBptxTemplate(const L1GtBptxTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtBptxTemplate::~L1GtBptxTemplate()
{
    // empty now
}

// assign operator
L1GtBptxTemplate& L1GtBptxTemplate::operator= (const L1GtBptxTemplate& cp)
{

    copy(cp);
    return *this;
}


void L1GtBptxTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtBptxTemplate print..." << std::endl;

    L1GtCondition::print(myCout);


    // reset to decimal output
    myCout << std::dec << std::endl;
}

void L1GtBptxTemplate::copy(const L1GtBptxTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtBptxTemplate& result)
{
    result.print(os);
    return os;

}


