/**
 * \class L1GlobalTriggerObjectMapRecord
 * 
 * 
 * Description: see header file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// system include files

#include <algorithm>

// user include files

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// methods

/// return the object map for the algorithm algoNameVal
const L1GlobalTriggerObjectMap* L1GlobalTriggerObjectMapRecord::getObjectMap(const std::string& algoNameVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoName() == algoNameVal) {
      return &((*itObj));
    }
  }

  // no algoName found, return zero pointer!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested algorithm name = " << algoNameVal << " does not exist in the trigger menu."
      << " Returning zero pointer for getObjectMap." << std::endl;

  return nullptr;
}

/// return the object map for the algorithm with bit number const int algoBitNumberVal
const L1GlobalTriggerObjectMap* L1GlobalTriggerObjectMapRecord::getObjectMap(const int algoBitNumberVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoBitNumber() == algoBitNumberVal) {
      return &((*itObj));
    }
  }

  // no algoBitNumberVal found, return zero pointer!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested algorithm with bit number = " << algoBitNumberVal
      << " does not exist in the trigger menu."
      << " Returning zero pointer for getObjectMap." << std::endl;

  return nullptr;
}

// return all the combinations passing the requirements imposed in condition condNameVal
// from algorithm algoNameVal
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(const std::string& algoNameVal,
                                                                                const std::string& condNameVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoName() == algoNameVal) {
      return itObj->getCombinationsInCond(condNameVal);
    }
  }

  // no (algoName, condName) found, return zero pointer!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested (algorithm name, condition name) = (" << algoNameVal << ", " << condNameVal
      << ") does not exist in the trigger menu."
      << " Returning zero pointer for getCombinationsInCond." << std::endl;

  return nullptr;
}

// return all the combinations passing the requirements imposed in condition condNameVal
// from algorithm with bit number algoBitNumberVal
const CombinationsInCond* L1GlobalTriggerObjectMapRecord::getCombinationsInCond(const int algoBitNumberVal,
                                                                                const std::string& condNameVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoBitNumber() == algoBitNumberVal) {
      return itObj->getCombinationsInCond(condNameVal);
    }
  }

  // no (algoBitNumber, condName) found, return zero pointer!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested (algorithm bit number, condition name) = (" << algoBitNumberVal << ", " << condNameVal
      << ") does not exist in the trigger menu."
      << " Returning zero pointer for getCombinationsInCond." << std::endl;

  return nullptr;
}

// return the result for the condition condNameVal
// from algorithm with name algoNameVal
bool L1GlobalTriggerObjectMapRecord::getConditionResult(const std::string& algoNameVal,
                                                        const std::string& condNameVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoName() == algoNameVal) {
      return itObj->getConditionResult(condNameVal);
    }
  }

  // no (algoName, condName) found, return false!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested (algorithm name, condition name) = (" << algoNameVal << ", " << condNameVal
      << ") does not exist in the trigger menu."
      << " Returning false for condition result! Unknown result, in fact!" << std::endl;

  return false;
}

// return the result for the condition condNameVal
// from algorithm with bit number algoBitNumberVal
bool L1GlobalTriggerObjectMapRecord::getConditionResult(const int algoBitNumberVal,
                                                        const std::string& condNameVal) const {
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itObj = m_gtObjectMap.begin();
       itObj != m_gtObjectMap.end();
       ++itObj) {
    if (itObj->algoBitNumber() == algoBitNumberVal) {
      return itObj->getConditionResult(condNameVal);
    }
  }

  // no (algoBitNumber, condName) found, return false!
  edm::LogError("L1GlobalTriggerObjectMapRecord")
      << " ERROR: The requested (algorithm bit number, condition name) = (" << algoBitNumberVal << ", " << condNameVal
      << ") does not exist in the trigger menu."
      << " Returning false for condition result! Unknown result, in fact!" << std::endl;

  return false;
}
