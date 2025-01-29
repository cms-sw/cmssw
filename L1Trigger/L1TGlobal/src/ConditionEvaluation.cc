/**
 * \class ConditionEvaluation
 *
 *
 * Description: Base class for evaluation of the L1 Global Trigger object templates.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include <ostream>

/// print condition
void l1t::ConditionEvaluation::print(std::ostream& myCout) const {
  myCout << "\n  ConditionEvaluation print...\n" << std::endl;
  myCout << "  Maximum number of objects in condition: " << m_condMaxNumberObjects << std::endl;
  myCout << "  Condition result:                       " << m_condLastResult << std::endl;

  std::ostringstream myCout1;
  for (size_t i1 = 0; i1 < m_combinationsInCond.size(); ++i1) {
    myCout1 << "( ";
    for (size_t i2 = 0; i2 < m_combinationsInCond[i1].size(); ++i2) {
      myCout1 << m_combinationsInCond[i1][i2].first << ":" << m_combinationsInCond[i1][i2].second << " ";
    }
    myCout1 << "); ";
  }

  myCout << "\n  List of combinations passing all requirements for this condition: \n  " << myCout1.str() << " \n"
         << std::endl;
}
