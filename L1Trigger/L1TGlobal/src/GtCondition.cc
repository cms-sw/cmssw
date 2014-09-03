/**
 * \class GtCondition
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
#include "L1Trigger/L1TGlobal/interface/GtCondition.h"

// system include files

// user include files

// forward declarations

// constructor
//    empty
GtCondition::GtCondition()
{

  m_condCategory = l1t::CondNull;
  m_condType = l1t::TypeNull;
  m_condChipNr = -1;
  m_condRelativeBx = 0;

    // the rest of private members are C++ initialized
}

//    constructor from condition name
GtCondition::GtCondition(const std::string& cName)
{
    m_condName = cName;

    m_condCategory = l1t::CondNull;
    m_condType = l1t::TypeNull;
    m_condChipNr = -1;
    m_condRelativeBx = 0;

}

//   constructor from condition name, category and type
GtCondition::GtCondition(const std::string& cName,
                             const l1t::GtConditionCategory& cCategory,
                             const l1t::GtConditionType& cType)
{

    m_condName = cName;
    m_condCategory = cCategory;
    m_condType = cType;

    m_condChipNr = -1;
    m_condRelativeBx = 0;

}



GtCondition::~GtCondition()
{
    // empty
}

// get number of trigger objects
const int GtCondition::nrObjects() const
{

    switch (m_condType) {

        case l1t::TypeNull:
        case TypeExternal:
        case TypeCastor:
        case TypeBptx: {
                return 0;
            }

            break;
        case l1t::Type1s: {
                return 1;
            }

            break;
        case l1t::Type2s:
        case l1t::Type2wsc:
        case l1t::Type2cor: {
                return 2;
            }

            break;
        case l1t::Type3s: {
                return 3;
            }

            break;
        case l1t::Type4s: {
                return 4;
            }

            break;
        case TypeETT:
        case TypeETM:
        case TypeHTT:
        case TypeHTM:
        case TypeJetCounts:
        case TypeHfBitCounts:
        case TypeHfRingEtSums: {
                return 1;
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
const bool GtCondition::wsc() const
{

    if (m_condType == l1t::Type2wsc) {
        return true;
    }

    return false;
}

// get logic flag for conditions, different type of trigger objects,
// and with spatial correlations
const bool GtCondition::corr() const
{

    if (m_condType == l1t::Type2cor) {
        return true;
    }

    return false;
}

// print condition
void GtCondition::print(std::ostream& myCout) const
{

    myCout << "\n  Condition name:     " << m_condName << std::endl;

    switch (m_condCategory) {
        case l1t::CondNull: {
                myCout << "  Condition category: " << "l1t::CondNull"
                << " - it means not defined!"
                << std::endl;
            }

            break;
        case l1t::CondMuon: {
                myCout << "  Condition category: " << "l1t::CondMuon"  << std::endl;
            }

            break;
        case l1t::CondCalo: {
                myCout << "  Condition category: " << "l1t::CondCalo"  << std::endl;
            }

            break;
        case CondEnergySum: {
                myCout << "  Condition category: " << "CondEnergySum"  << std::endl;
            }

            break;
        case CondJetCounts: {
                myCout << "  Condition category: " << "CondJetCounts"  << std::endl;
            }

            break;
        case l1t::CondCorrelation: {
                myCout << "  Condition category: " << "l1t::CondCorrelation"  << std::endl;
            }

            break;
        case CondCastor: {
                myCout << "  Condition category: " << "CondCastor"  << std::endl;
            }

            break;
        case CondHfBitCounts: {
                myCout << "  Condition category: " << "CondHfBitCounts"  << std::endl;
            }

            break;
        case CondHfRingEtSums: {
                myCout << "  Condition category: " << "CondHfRingEtSums"  << std::endl;
            }

            break;
        case CondBptx: {
                myCout << "  Condition category: " << "CondBptx"  << std::endl;
            }

            break;
        case CondExternal: {
                myCout << "  Condition category: " << "CondExternal"  << std::endl;
            }

            break;
        default: {
                myCout << "  Condition category: " << m_condCategory
                << "  - no such category defined. Check l1t::GtConditionCategory enum."
                << std::endl;

            }
            break;
    }

    switch (m_condType) {

        case l1t::TypeNull: {
                myCout << "  Condition type:     " << "l1t::TypeNull"
                << " - it means not defined!"
                << std::endl;
            }

            break;
        case l1t::Type1s: {
                myCout << "  Condition type:     " << "l1t::Type1s"  << std::endl;
            }

            break;
        case l1t::Type2s: {
                myCout << "  Condition type:     " << "l1t::Type2s"  << std::endl;
            }

            break;
        case l1t::Type2wsc: {
                myCout << "  Condition type:     " << "l1t::Type2wsc"  << std::endl;
            }

            break;
        case l1t::Type2cor: {
                myCout << "  Condition type:     " << "l1t::Type2cor"  << std::endl;
            }

            break;
        case l1t::Type3s: {
                myCout << "  Condition type:     " << "l1t::Type3s"  << std::endl;
            }

            break;
        case l1t::Type4s: {
                myCout << "  Condition type:     " << "l1t::Type4s"  << std::endl;
            }

            break;
        case TypeETM: {
                myCout << "  Condition type:     " << "TypeETM"  << std::endl;
            }

            break;
        case TypeETT: {
                myCout << "  Condition type:     " << "TypeETT"  << std::endl;
            }

            break;
        case TypeHTT: {
                myCout << "  Condition type:     " << "TypeHTT"  << std::endl;
            }

            break;
        case TypeHTM: {
                myCout << "  Condition type:     " << "TypeHTM"  << std::endl;
            }

            break;
        case TypeJetCounts: {
                myCout << "  Condition type:     " << "TypeJetCounts"  << std::endl;
            }

            break;
        case TypeCastor: {
                myCout << "  Condition type:     " << "TypeCastor"  << std::endl;
            }

            break;
        case TypeHfBitCounts: {
                myCout << "  Condition type:     " << "TypeHfBitCounts"  << std::endl;
            }

            break;
        case TypeHfRingEtSums: {
                myCout << "  Condition type:     " << "TypeHfRingEtSums"  << std::endl;
            }

            break;
        case TypeBptx: {
                myCout << "  Condition type:     " << "TypeBptx"  << std::endl;
            }

            break;
        case TypeExternal: {
                myCout << "  Condition type:     " << "TypeExternal"  << std::endl;
            }

            break;
        default: {
                myCout << "  Condition type:     " << m_condType
                << " - no such type defined. Check l1t::GtConditionType enum."
                << std::endl;
            }
            break;
    }


    myCout << "  Object types:      ";

    for (unsigned int i = 0; i < m_objectType.size(); ++i) {

        switch (m_objectType[i]) {
            case Mu: {
                    myCout << " Mu ";
                }

                break;
            case NoIsoEG: {
                    myCout << " NoIsoEG ";
                }

                break;
            case IsoEG: {
                    myCout << " IsoEG ";
                }

                break;
            case CenJet: {
                    myCout << " CenJet ";
                }

                break;
            case ForJet: {
                    myCout << " ForJet ";
                }

                break;
            case TauJet: {
                    myCout << " TauJet ";
                }

                break;
            case ETM: {
                    myCout << " ETM ";
                }

                break;
            case ETT: {
                    myCout << " ETT ";
                }

                break;
            case HTT: {
                    myCout << " HTT ";
                }

                break;
            case HTM: {
                    myCout << " HTM ";
                }

                break;
            case JetCounts: {
                    myCout << " JetCounts ";
                }

                break;
            case HfBitCounts: {
                    myCout << " HfBitCounts ";
                }

                break;
            case HfRingEtSums: {
                    myCout << " HfRingEtSums ";
                }

                break;
            case BPTX: {
                    myCout << " BPTX ";
                }

                break;
            case GtExternal: {
                    myCout << " GtExternal ";
                }

                break;
            default: {
                    myCout << " Unknown type " << m_objectType[i];
                }
                break;
        }
    }

    myCout << std::endl;

    myCout << "  \" >= \" flag:        " << m_condGEq << std::endl;

    myCout << "  Condition chip:     " << m_condChipNr;

    if (m_condChipNr < 0) {
        myCout << "   - not properly initialized! ";
    }

    myCout << std::endl;

    myCout << "  Relative BX:     " << m_condRelativeBx << std::endl;

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const GtCondition& result)
{
    result.print(os);
    return os;

}

