/**
 * \class GlobalObjectMap
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

#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>

/// return all the combinations passing the requirements imposed in condition condNameVal
const CombinationsWithBxInCond* GlobalObjectMap::getCombinationsInCond(const std::string& condNameVal) const {
  for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {
    if ((m_operandTokenVector[i]).tokenName == condNameVal) {
      return &(m_combinationWithBxVector.at((m_operandTokenVector[i]).tokenNumber));
    }
  }

  // return a null address - should not arrive here
  edm::LogError("GlobalObjectMap") << "\n\n  ERROR: The requested condition with tokenName = " << condNameVal
                                   << "\n  does not exists in the operand token vector."
                                   << "\n  Returning zero pointer for getCombinationsInCond\n\n";

  return nullptr;
}

/// return all the combinations passing the requirements imposed in condition condNumberVal
const CombinationsWithBxInCond* GlobalObjectMap::getCombinationsInCond(const int condNumberVal) const {
  for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {
    if ((m_operandTokenVector[i]).tokenNumber == condNumberVal) {
      return &(m_combinationWithBxVector.at((m_operandTokenVector[i]).tokenNumber));
    }
  }

  // return a null address - should not arrive here
  edm::LogError("GlobalObjectMap") << "\n\n  ERROR: The requested condition with tokenNumber = " << condNumberVal
                                   << "\n  does not exists in the operand token vector."
                                   << "\n  Returning zero pointer for getCombinationsInCond\n\n";

  return nullptr;
}

/// return the result for the condition condNameVal
const bool GlobalObjectMap::getConditionResult(const std::string& condNameVal) const {
  for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {
    if ((m_operandTokenVector[i]).tokenName == condNameVal) {
      return (m_operandTokenVector[i]).tokenResult;
    }
  }

  // return false - should not arrive here
  edm::LogError("GlobalObjectMap") << "\n\n  ERROR: The requested condition with name = " << condNameVal
                                   << "\n  does not exists in the operand token vector."
                                   << "\n  Returning false for getConditionResult\n\n";

  return false;
}

void GlobalObjectMap::reset() {
  // name of the algorithm
  m_algoName.clear();

  // bit number for algorithm
  m_algoBitNumber = -1;

  // GTL result of the algorithm
  m_algoGtlResult = false;

  // vector of operand tokens for an algorithm
  m_operandTokenVector.clear();

  // vector of combinations for all conditions in an algorithm
  m_combinationWithBxVector.clear();
}

void GlobalObjectMap::print(std::ostream& myCout) const {
  myCout << "GlobalObjectMap: print " << std::endl;

  myCout << "  Algorithm name: " << m_algoName << std::endl;
  myCout << "    Bit number: " << m_algoBitNumber << std::endl;
  myCout << "    GTL result: " << m_algoGtlResult << std::endl;

  int operandTokenVectorSize = m_operandTokenVector.size();

  myCout << "    Operand token vector size: " << operandTokenVectorSize;

  if (operandTokenVectorSize == 0) {
    myCout << "   - not properly initialized! " << std::endl;
  } else {
    myCout << std::endl;

    for (int i = 0; i < operandTokenVectorSize; ++i) {
      myCout << "      " << std::setw(5) << (m_operandTokenVector[i]).tokenNumber << "\t" << std::setw(25)
             << (m_operandTokenVector[i]).tokenName << "\t" << (m_operandTokenVector[i]).tokenResult << std::endl;
    }
  }

  myCout << "    CombinationWithBxVector size: " << m_combinationWithBxVector.size() << std::endl;

  myCout << "  conditions: " << std::endl;

  for (size_t i1 = 0; i1 < m_combinationWithBxVector.size(); ++i1) {
    auto const& condName = m_operandTokenVector[i1].tokenName;
    auto const condResult = m_operandTokenVector[i1].tokenResult;
    myCout << "    Condition " << condName << " evaluated to " << condResult << std::endl;
    myCout << "    List of combinations passing all requirements for this condition:" << std::endl;
    myCout << "    ";

    if (m_combinationWithBxVector[i1].empty()) {
      myCout << "(none)";
    } else {
      for (size_t i2 = 0; i2 < m_combinationWithBxVector[i1].size(); ++i2) {
        myCout << "( ";
        for (size_t i3 = 0; i3 < m_combinationWithBxVector[i1][i2].size(); ++i3) {
          myCout << m_combinationWithBxVector[i1][i2][i3].first << ":";
          myCout << m_combinationWithBxVector[i1][i2][i3].second << " ";
        }
        myCout << "); ";
      }
    }
    myCout << "\n\n";
  }
}
