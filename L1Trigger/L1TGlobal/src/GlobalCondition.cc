/**
 * \class GlobalCondition
 *
 *
 * Description: base class for L1 Global Trigger object templates (condition).
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vladimir Rekovic,   Brian Winer, OSU   Vasile Mihai Ghete - HEPHY Vienna
 *          Elisa Fontanesi - extended for three-body correlation conditions
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// system include files

// user include files

// forward declarations

// constructor
//    empty
GlobalCondition::GlobalCondition() {
  m_condCategory = l1t::CondNull;
  m_condType = l1t::TypeNull;
  m_condChipNr = -1;
  m_condRelativeBx = 0;

  // the rest of private members are C++ initialized
}

//    constructor from condition name
GlobalCondition::GlobalCondition(const std::string& cName) {
  m_condName = cName;

  m_condCategory = l1t::CondNull;
  m_condType = l1t::TypeNull;
  m_condChipNr = -1;
  m_condRelativeBx = 0;
}

//   constructor from condition name, category and type
GlobalCondition::GlobalCondition(const std::string& cName,
                                 const l1t::GtConditionCategory& cCategory,
                                 const l1t::GtConditionType& cType) {
  m_condName = cName;
  m_condCategory = cCategory;
  m_condType = cType;

  m_condChipNr = -1;
  m_condRelativeBx = 0;
}

GlobalCondition::~GlobalCondition() {
  // empty
}

// get number of trigger objects
const int GlobalCondition::nrObjects() const {
  switch (m_condType) {
    case l1t::TypeNull:
    case l1t::TypeExternal: {
      return 0;
    }

    break;
    case l1t::TypeCent0:
    case l1t::TypeCent1:
    case l1t::TypeCent2:
    case l1t::TypeCent3:
    case l1t::TypeCent4:
    case l1t::TypeCent5:
    case l1t::TypeCent6:
    case l1t::TypeCent7: {
      return 0;
    }

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
    case l1t::Type2corWithOverlapRemoval: {
      return 3;
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
    case l1t::TypeETT:
    case l1t::TypeETM:
    case l1t::TypeHTT:
    case l1t::TypeHTM:
    case l1t::TypeETMHF:
    case l1t::TypeTowerCount:
    case l1t::TypeMinBiasHFP0:
    case l1t::TypeMinBiasHFM0:
    case l1t::TypeMinBiasHFP1:
    case l1t::TypeMinBiasHFM1:
    case l1t::TypeZDCP:
    case l1t::TypeZDCM:
    case l1t::TypeAXOL1TL:
    case l1t::TypeAsymEt:
    case l1t::TypeAsymHt:
    case l1t::TypeAsymEtHF:
    case l1t::TypeAsymHtHF: {
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
const bool GlobalCondition::wsc() const {
  if (m_condType == l1t::Type2wsc) {
    return true;
  }

  return false;
}

// get logic flag for conditions, different type of trigger objects,
// and with spatial correlations
const bool GlobalCondition::corr() const {
  if (m_condType == l1t::Type2cor) {
    return true;
  }

  return false;
}

// get logic flag for conditions, trigger objects are muons
const bool GlobalCondition::corrThree() const {
  if (m_condType == l1t::Type3s) {
    return true;
  }

  return false;
}

// print condition
void GlobalCondition::print(std::ostream& myCout) const {
  myCout << "\n  Condition name:     " << m_condName << std::endl;

  switch (m_condCategory) {
    case l1t::CondNull: {
      myCout << "  Condition category: "
             << "l1t::CondNull"
             << " - it means not defined!" << std::endl;
    }

    break;
    case l1t::CondMuon: {
      myCout << "  Condition category: "
             << "l1t::CondMuon" << std::endl;
    }

    break;
    case l1t::CondMuonShower: {
      myCout << "  Condition category: "
             << "l1t::CondMuonShower" << std::endl;
    }

    break;
    case l1t::CondCalo: {
      myCout << "  Condition category: "
             << "l1t::CondCalo" << std::endl;
    }

    break;
    case l1t::CondEnergySum: {
      myCout << "  Condition category: "
             << "CondEnergySum" << std::endl;
    }

    break;
    case l1t::CondEnergySumZdc: {
      myCout << "  Condition category: "
             << "CondEnergySumZdc" << std::endl;
    }

    break;
    case l1t::CondAXOL1TL: {
      myCout << "  Condition category: "
             << "CondAXOL1TL" << std::endl;
    }

    break;
    case l1t::CondCorrelation: {
      myCout << "  Condition category: "
             << "CondCorrelation" << std::endl;
    }

    break;
    case l1t::CondCorrelationThreeBody: {
      myCout << "  Condition category: "
             << "CondCorrelationThreeBody" << std::endl;
    }

    break;
    case l1t::CondCorrelationWithOverlapRemoval: {
      myCout << "  Condition category: "
             << "CondCorrelationWithOverlapRemoval" << std::endl;
    }

    break;
    case l1t::CondExternal: {
      myCout << "  Condition category: "
             << "CondExternal" << std::endl;
    }

    break;
    default: {
      myCout << "  Condition category: " << m_condCategory
             << "  - no such category defined. Check l1t::GtConditionCategory enum." << std::endl;

    } break;
  }

  switch (m_condType) {
    case l1t::TypeNull: {
      myCout << "  Condition type:     "
             << "l1t::TypeNull"
             << " - it means not defined!" << std::endl;
    }

    break;
    case l1t::Type1s: {
      myCout << "  Condition type:     "
             << "l1t::Type1s" << std::endl;
    }

    break;
    case l1t::Type2s: {
      myCout << "  Condition type:     "
             << "l1t::Type2s" << std::endl;
    }

    break;
    case l1t::Type2corWithOverlapRemoval: {
      myCout << "  Condition type:     "
             << "l1t::Type2corWithOverlapRemoval" << std::endl;
    }

    break;
    case l1t::Type2wsc: {
      myCout << "  Condition type:     "
             << "l1t::Type2wsc" << std::endl;
    }

    break;
    case l1t::Type2cor: {
      myCout << "  Condition type:     "
             << "l1t::Type2cor" << std::endl;
    }

    break;
    case l1t::Type3s: {
      myCout << "  Condition type:     "
             << "l1t::Type3s" << std::endl;
    }

    break;
    case l1t::Type4s: {
      myCout << "  Condition type:     "
             << "l1t::Type4s" << std::endl;
    }

    break;
    case l1t::TypeETM: {
      myCout << "  Condition type:     "
             << "TypeETM" << std::endl;
    }

    break;
    case l1t::TypeETT: {
      myCout << "  Condition type:     "
             << "TypeETT" << std::endl;
    }

    break;
    case l1t::TypeHTT: {
      myCout << "  Condition type:     "
             << "TypeHTT" << std::endl;
    }

    break;
    case l1t::TypeHTM: {
      myCout << "  Condition type:     "
             << "TypeHTM" << std::endl;
    }

    break;
    case l1t::TypeETMHF: {
      myCout << "  Condition type:     "
             << "TypeETMHF" << std::endl;
    }

    break;
    case l1t::TypeTowerCount: {
      myCout << "  Condition type:     "
             << "TypeTowerCount" << std::endl;
    }

    break;
    case l1t::TypeMinBiasHFP0: {
      myCout << "  Condition type:     "
             << "TypeMinBiasHFP0" << std::endl;
    }

    break;
    case l1t::TypeMinBiasHFM0: {
      myCout << "  Condition type:     "
             << "TypeMinBiasHFM0" << std::endl;
    }

    break;
    case l1t::TypeMinBiasHFP1: {
      myCout << "  Condition type:     "
             << "TypeMinBiasHFP1" << std::endl;
    }

    break;
    case l1t::TypeMinBiasHFM1: {
      myCout << "  Condition type:     "
             << "TypeMinBiasHFM1" << std::endl;
    }

    break;
    case l1t::TypeETTem: {
      myCout << "  Condition type:     "
             << "TypeETTem" << std::endl;
    }

    break;
    case l1t::TypeAsymEt: {
      myCout << "  Condition type:     "
             << "TypeAsymEt" << std::endl;
    }

    break;
    case l1t::TypeAsymHt: {
      myCout << "  Condition type:     "
             << "TypeAsymHt" << std::endl;
    }

    break;
    case l1t::TypeAsymEtHF: {
      myCout << "  Condition type:     "
             << "TypeAsymEtHF" << std::endl;
    }

    break;
    case l1t::TypeAsymHtHF: {
      myCout << "  Condition type:     "
             << "TypeAsymHtHF" << std::endl;
    }

    break;
    case l1t::TypeCent0: {
      myCout << "  Condition type:     "
             << "TypeCent0" << std::endl;
    }

    break;
    case l1t::TypeCent1: {
      myCout << "  Condition type:     "
             << "TypeCent1" << std::endl;
    }

    break;
    case l1t::TypeCent2: {
      myCout << "  Condition type:     "
             << "TypeCent2" << std::endl;
    }

    break;
    case l1t::TypeCent3: {
      myCout << "  Condition type:     "
             << "TypeCent3" << std::endl;
    }

    break;
    case l1t::TypeCent4: {
      myCout << "  Condition type:     "
             << "TypeCent4" << std::endl;
    }

    break;
    case l1t::TypeCent5: {
      myCout << "  Condition type:     "
             << "TypeCent5" << std::endl;
    }

    break;
    case l1t::TypeCent6: {
      myCout << "  Condition type:     "
             << "TypeCent6" << std::endl;
    }

    break;
    case l1t::TypeCent7: {
      myCout << "  Condition type:     "
             << "TypeCent7" << std::endl;
    }

    break;
    case l1t::TypeExternal: {
      myCout << "  Condition type:     "
             << "TypeExternal" << std::endl;
    }

    break;
    default: {
      myCout << "  Condition type:     " << m_condType << " - no such type defined. Check l1t::GtConditionType enum."
             << std::endl;
    } break;
  }

  myCout << "  Object types:      ";

  for (unsigned int i = 0; i < m_objectType.size(); ++i) {
    switch (m_objectType[i]) {
      case l1t::gtMu: {
        myCout << " Mu ";
      }

      break;
      case l1t::gtMuShower: {
        myCout << " MuShower ";
      }

      break;
      case l1t::gtEG: {
        myCout << " EG ";
      }

      break;

      case l1t::gtJet: {
        myCout << " Jet ";
      }

      break;

      case l1t::gtTau: {
        myCout << " Tau ";
      }

      break;
      case l1t::gtETM: {
        myCout << " ETM ";
      }

      break;
      case l1t::gtETT: {
        myCout << " ETT ";
      }

      break;
      case l1t::gtHTT: {
        myCout << " HTT ";
      }

      break;
      case l1t::gtHTM: {
        myCout << " HTM ";
      }

      break;

      case l1t::gtETMHF: {
        myCout << " ETMHF ";
      }

      break;
      case l1t::gtTowerCount: {
        myCout << " TowerCount ";
      }

      break;
      case l1t::gtMinBiasHFP0: {
        myCout << " MinBiasHFP0 ";
      }

      break;
      case l1t::gtMinBiasHFM0: {
        myCout << " MinBiasHFM0 ";
      }

      break;
      case l1t::gtMinBiasHFP1: {
        myCout << " MinBiasHFP1 ";
      }

      break;
      case l1t::gtMinBiasHFM1: {
        myCout << " MinBiasHFM1 ";
      }

      break;
      case l1t::gtETTem: {
        myCout << " ETTem ";
      }

      break;
      case l1t::gtAsymmetryEt: {
        myCout << " AsymmetryEt ";
      }

      break;
      case l1t::gtAsymmetryHt: {
        myCout << " AsymmetryHt ";
      }

      break;
      case l1t::gtAsymmetryEtHF: {
        myCout << " AsymmetryEtHF ";
      }

      break;
      case l1t::gtAsymmetryHtHF: {
        myCout << " AsymmetryHtHF ";
      }

      break;
      case l1t::gtCentrality0: {
        myCout << " Centrality0 ";
      }

      break;
      case l1t::gtCentrality1: {
        myCout << " Centrality1 ";
      }

      break;
      case l1t::gtCentrality2: {
        myCout << " Centrality2 ";
      }

      break;
      case l1t::gtCentrality3: {
        myCout << " Centrality3 ";
      }

      break;
      case l1t::gtCentrality4: {
        myCout << " Centrality4 ";
      }

      break;
      case l1t::gtCentrality5: {
        myCout << " Centrality5 ";
      }

      break;
      case l1t::gtCentrality6: {
        myCout << " Centrality6 ";
      }

      break;
      case l1t::gtCentrality7: {
        myCout << " Centrality7 ";
      }

      break;

      case l1t::gtExternal: {
        myCout << " External ";
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

  myCout << "  Relative BX:     " << m_condRelativeBx << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const GlobalCondition& result) {
  result.print(os);
  return os;
}
