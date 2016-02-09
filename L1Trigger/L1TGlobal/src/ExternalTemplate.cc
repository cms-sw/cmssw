/**
 * \class ExternalTemplate
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
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// constructors
ExternalTemplate::ExternalTemplate()
        : GtCondition()
{

    m_condCategory = l1t::CondExternal;

}

ExternalTemplate::ExternalTemplate(const std::string& cName)
        : GtCondition(cName)
{

    m_condCategory = l1t::CondExternal;

}

ExternalTemplate::ExternalTemplate(const std::string& cName, const l1t::GtConditionType& cType)
        : GtCondition(cName, l1t::CondExternal, cType)
{

    m_condCategory = l1t::CondExternal;

}

// copy constructor
ExternalTemplate::ExternalTemplate(const ExternalTemplate& cp)
        : GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
ExternalTemplate::~ExternalTemplate()
{
    // empty now
}

// assign operator
ExternalTemplate& ExternalTemplate::operator= (const ExternalTemplate& cp)
{

    copy(cp);
    return *this;
}


void ExternalTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  ExternalTemplate print..." << std::endl;

    GtCondition::print(myCout);


    myCout << "  External Channel " << m_extChannel << std::endl;

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void ExternalTemplate::copy(const ExternalTemplate& cp)
{

    m_condName       = cp.condName();
    m_condCategory   = cp.condCategory();
    m_condType       = cp.condType();
    m_objectType     = cp.objectType();
    m_condGEq        = cp.condGEq();
    m_condChipNr     = cp.condChipNr();
    m_condRelativeBx = cp.condRelativeBx();
    m_extChannel     = cp.extChannel();

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const ExternalTemplate& result)
{
    result.print(os);
    return os;

}


