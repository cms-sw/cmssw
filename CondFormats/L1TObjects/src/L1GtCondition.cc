/**
 * \class L1GtCondition
 * 
 * 
 * Description: base class for L1 Global Trigger object templates (condition).  
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
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"

// system include files
// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// constructor
//    empty
L1GtCondition::L1GtCondition()
{
    // empty
}

//    constructor from condition name
L1GtCondition::L1GtCondition(const std::string& cName)
{
    m_condName = cName;
}

//   constructor from condition name and type
L1GtCondition::L1GtCondition(const std::string& cName, const L1GtConditionType& cType)
{

    m_condName = cName;
    m_condType = cType;

}



L1GtCondition::~L1GtCondition()
{
    // empty
}

// get number of trigger objects
const int L1GtCondition::nrObjects() const
{

    switch (m_condType) {
        case Type1s: {
                return 1;
            }

            break;
        case Type2s:
        case Type2wsc:
        case Type2cor: {
                return 2;
            }

            break;
        case Type3s: {
                return 3;
            }

            break;
        case Type4s: {
                return 4;
            }

            break;
        default: {
                // TODO no such type, throw exception?
                return 0;
            }
            break;
    }

}

// get logic flag for conditions, same type of trigger objects,
// and with spatial correlations
const bool L1GtCondition::wsc() const
{

    if (m_condType == Type2wsc) {
        return true;
    }

    return false;
}

// get logic flag for conditions, different type of trigger objects,
// and with spatial correlations
const bool L1GtCondition::corr() const
{

    if (m_condType == Type2cor) {
        return true;
    }

    return false;
}

