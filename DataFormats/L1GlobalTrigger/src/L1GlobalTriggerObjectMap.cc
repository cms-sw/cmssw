/**
 * \class L1GlobalTriggerObjectMap
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

// system include files
#include <iostream>
#include <iterator>

#include <vector>
#include <string>

#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

// forward declarations


// constructor(s)
L1GlobalTriggerObjectMap::L1GlobalTriggerObjectMap()
{}

// destructor
L1GlobalTriggerObjectMap::~L1GlobalTriggerObjectMap()
{}

// methods

void L1GlobalTriggerObjectMap::reset()
{

    // name of the algorithm
    m_algoName = "";

    // bit number for algorithm
    m_algoBitNumber = -1;

    // GTL result of the algorithm
    m_algoGtlResult = false;

    // logical expression for the algorithm
    m_algoLogicalExpression = "";

    // numerical expression for the algorithm
    // (logical expression with conditions replaced with the actual values)
    m_algoNumericalExpression = "";

    // vector of combinations for all conditions in an algorithm
    m_combinationVector.clear();

}

void L1GlobalTriggerObjectMap::print(std::ostream& myCout)
{

    myCout << "L1GlobalTriggerObjectMap: print " << std::endl;

    myCout << "  Algorithm name: " << m_algoName << std::endl;
    myCout << "    Bit number: " << m_algoBitNumber << std::endl;
    myCout << "    GTL result: " << m_algoGtlResult << std::endl;
    myCout << "    Logical Expression: " << m_algoLogicalExpression << std::endl;
    myCout << "    Numerical Expression: " << m_algoNumericalExpression << std::endl;
    myCout << "    CombinationVector size: " << m_combinationVector.size() << std::endl;

    myCout << "  conditions: "  << std::endl;

    std::vector<CombinationsInCond>::const_iterator itVVV;
    for(itVVV  = m_combinationVector.begin(); itVVV != m_combinationVector.end(); itVVV++) {

        myCout << "    Condition name: " << std::endl;
        myCout << "    List of combinations passing all requirements for this condition:"
        << std::endl;

        myCout << "    ";

        if ((*itVVV).size() == 0) {
            myCout << "(none)";
        } else {

            CombinationsInCond::const_iterator itVV;
            for(itVV  = (*itVVV).begin(); itVV != (*itVVV).end(); itVV++) {

                myCout << "( ";

                std::copy((*itVV).begin(), (*itVV).end(),
                          std::ostream_iterator<int> (myCout, " "));

                myCout << "); ";

            }

        }

        myCout << "\n";
    }
}

