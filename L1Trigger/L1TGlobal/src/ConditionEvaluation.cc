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

// this class header
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

// system include files
#include <iostream>
#include <iomanip>
#include <iterator>

// user include files

//   base class

//

// methods

/// print condition
void l1t::ConditionEvaluation::print(std::ostream& myCout) const {

    myCout << "\n  ConditionEvaluation print...\n" << std::endl;
    myCout << "  Maximum number of objects in condition: " << m_condMaxNumberObjects << std::endl;
    myCout << "  Condition result:                       " << m_condLastResult << std::endl;

    CombinationsInCond::const_iterator itVV;
    std::ostringstream myCout1;

    for (itVV = (m_combinationsInCond).begin(); itVV != (m_combinationsInCond).end(); itVV++) {

        myCout1 << "( ";

        std::copy((*itVV).begin(), (*itVV).end(), std::ostream_iterator<int> (myCout1, " "));

        myCout1 << "); ";

    }

    myCout
    << "\n  List of combinations passing all requirements for this condition: \n  "
    << myCout1.str()
    << " \n"
    << std::endl;

}

