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

// forward declarations

// constructor
//    empty
L1GtCondition::L1GtCondition() {
  m_condCategory = CondNull;
  m_condType = TypeNull;
  m_condChipNr = -1;
  m_condGEq = false;

  // the rest of private members are C++ initialized
}

//    constructor from condition name
L1GtCondition::L1GtCondition(const std::string& cName) {
  m_condName = cName;

  m_condCategory = CondNull;
  m_condType = TypeNull;
  m_condChipNr = -1;
}

//   constructor from condition name, category and type
L1GtCondition::L1GtCondition(const std::string& cName,
                             const L1GtConditionCategory& cCategory,
                             const L1GtConditionType& cType) {
  m_condName = cName;
  m_condCategory = cCategory;
  m_condType = cType;

  m_condChipNr = -1;
}

L1GtCondition::~L1GtCondition() {
  // empty
}

// get number of trigger objects
const int L1GtCondition::nrObjects() const {
  switch (m_condType) {
    case TypeNull:
    case TypeExternal:
    case TypeCastor:
    case TypeBptx: {
      return 0;
    }

    break;
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
    } break;
  }
}

// get logic flag for conditions, same type of trigger objects,
// and with spatial correlations
const bool L1GtCondition::wsc() const {
  if (m_condType == Type2wsc) {
    return true;
  }

  return false;
}

// get logic flag for conditions, different type of trigger objects,
// and with spatial correlations
const bool L1GtCondition::corr() const {
  if (m_condType == Type2cor) {
    return true;
  }

  return false;
}

// print condition
void L1GtCondition::print(std::ostream& myCout) const {
  myCout << "\n  Condition name:     " << m_condName << std::endl;

  switch (m_condCategory) {
    case CondNull: {
      myCout << "  Condition category: "
             << "CondNull"
             << " - it means not defined!" << std::endl;
    }

    break;
    case CondMuon: {
      myCout << "  Condition category: "
             << "CondMuon" << std::endl;
    }

    break;
    case CondCalo: {
      myCout << "  Condition category: "
             << "CondCalo" << std::endl;
    }

    break;
    case CondEnergySum: {
      myCout << "  Condition category: "
             << "CondEnergySum" << std::endl;
    }

    break;
    case CondJetCounts: {
      myCout << "  Condition category: "
             << "CondJetCounts" << std::endl;
    }

    break;
    case CondCorrelation: {
      myCout << "  Condition category: "
             << "CondCorrelation" << std::endl;
    }

    break;
    case CondCastor: {
      myCout << "  Condition category: "
             << "CondCastor" << std::endl;
    }

    break;
    case CondHfBitCounts: {
      myCout << "  Condition category: "
             << "CondHfBitCounts" << std::endl;
    }

    break;
    case CondHfRingEtSums: {
      myCout << "  Condition category: "
             << "CondHfRingEtSums" << std::endl;
    }

    break;
    case CondBptx: {
      myCout << "  Condition category: "
             << "CondBptx" << std::endl;
    }

    break;
    case CondExternal: {
      myCout << "  Condition category: "
             << "CondExternal" << std::endl;
    }

    break;
    default: {
      myCout << "  Condition category: " << m_condCategory
             << "  - no such category defined. Check L1GtConditionCategory enum." << std::endl;

    } break;
  }

  switch (m_condType) {
    case TypeNull: {
      myCout << "  Condition type:     "
             << "TypeNull"
             << " - it means not defined!" << std::endl;
    }

    break;
    case Type1s: {
      myCout << "  Condition type:     "
             << "Type1s" << std::endl;
    }

    break;
    case Type2s: {
      myCout << "  Condition type:     "
             << "Type2s" << std::endl;
    }

    break;
    case Type2wsc: {
      myCout << "  Condition type:     "
             << "Type2wsc" << std::endl;
    }

    break;
    case Type2cor: {
      myCout << "  Condition type:     "
             << "Type2cor" << std::endl;
    }

    break;
    case Type3s: {
      myCout << "  Condition type:     "
             << "Type3s" << std::endl;
    }

    break;
    case Type4s: {
      myCout << "  Condition type:     "
             << "Type4s" << std::endl;
    }

    break;
    case TypeETM: {
      myCout << "  Condition type:     "
             << "TypeETM" << std::endl;
    }

    break;
    case TypeETT: {
      myCout << "  Condition type:     "
             << "TypeETT" << std::endl;
    }

    break;
    case TypeHTT: {
      myCout << "  Condition type:     "
             << "TypeHTT" << std::endl;
    }

    break;
    case TypeHTM: {
      myCout << "  Condition type:     "
             << "TypeHTM" << std::endl;
    }

    break;
    case TypeJetCounts: {
      myCout << "  Condition type:     "
             << "TypeJetCounts" << std::endl;
    }

    break;
    case TypeCastor: {
      myCout << "  Condition type:     "
             << "TypeCastor" << std::endl;
    }

    break;
    case TypeHfBitCounts: {
      myCout << "  Condition type:     "
             << "TypeHfBitCounts" << std::endl;
    }

    break;
    case TypeHfRingEtSums: {
      myCout << "  Condition type:     "
             << "TypeHfRingEtSums" << std::endl;
    }

    break;
    case TypeBptx: {
      myCout << "  Condition type:     "
             << "TypeBptx" << std::endl;
    }

    break;
    case TypeExternal: {
      myCout << "  Condition type:     "
             << "TypeExternal" << std::endl;
    }

    break;
    default: {
      myCout << "  Condition type:     " << m_condType << " - no such type defined. Check L1GtConditionType enum."
             << std::endl;
    } break;
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
      } break;
    }
  }

  myCout << std::endl;

  myCout << "  \" >= \" flag:        " << m_condGEq << std::endl;

  myCout << "  Condition chip:     " << m_condChipNr;

  if (m_condChipNr < 0) {
    myCout << "   - not properly initialized! ";
  }

  myCout << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtCondition& result) {
  result.print(os);
  return os;
}
