/**
 * \class L1GtConditionEvaluation
 *
 *
 * Description: Base class for evaluation of the L1 Global Trigger object templates.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// system include files
#include <iostream>
#include <iomanip>
#include <iterator>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtConditionEvaluation::L1GtConditionEvaluation() :
    m_condMaxNumberObjects(0),
    m_condLastResult(false),
    m_combinationsInCond(new CombinationsInCond),
    m_verbosity(0)

{

    // empty

}

// copy constructor
L1GtConditionEvaluation::L1GtConditionEvaluation(L1GtConditionEvaluation& cp) {

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

// destructor
L1GtConditionEvaluation::~L1GtConditionEvaluation() {

    delete m_combinationsInCond;

}

// methods

/// print condition
void L1GtConditionEvaluation::print(std::ostream& myCout) const {

    myCout << "\n  L1GtConditionEvaluation print...\n" << std::endl;
    myCout << "  Maximum number of objects in condition: " << m_condMaxNumberObjects << std::endl;
    myCout << "  Condition result:                       " << m_condLastResult << std::endl;

    CombinationsInCond::const_iterator itVV;
    std::ostringstream myCout1;

    for (itVV = (*m_combinationsInCond).begin(); itVV != (*m_combinationsInCond).end(); itVV++) {

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

