/**
 * \class L1GlobalTriggerObjectMap
 * 
 * 
 * Description: see header file.  
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

// system include files
#include <iostream>
#include <iterator>

#include <vector>
#include <string>

#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

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

    std::string emptyString;

    // name of the algorithm
    m_algoName = emptyString;

    // bit number for algorithm
    m_algoBitNumber = -1;

    // GTL result of the algorithm
    m_algoGtlResult = false;

    // logical expression for the algorithm
    m_algoLogicalExpression = emptyString;

    // numerical expression for the algorithm
    // (logical expression with conditions replaced with the actual values)
    m_algoNumericalExpression = emptyString;

    // vector of combinations for all conditions in an algorithm
    m_combinationVector.clear();

    m_objectTypeVector.clear();


}

void L1GlobalTriggerObjectMap::print(std::ostream& myCout) const
{

    myCout << "L1GlobalTriggerObjectMap: print " << std::endl;

    myCout << "  Algorithm name: " << m_algoName << std::endl;
    myCout << "    Bit number: " << m_algoBitNumber << std::endl;
    myCout << "    GTL result: " << m_algoGtlResult << std::endl;
    myCout << "    Logical expression: '" << m_algoLogicalExpression << "'" << std::endl;
    myCout << "    Numerical expression: '" << m_algoNumericalExpression << "'" << std::endl;
    myCout << "    CombinationVector size: " << m_combinationVector.size() << std::endl;
    myCout << "    ObjectTypeVector size: " << m_objectTypeVector.size() << std::endl;

    myCout << "  conditions: "  << std::endl;

    std::vector<CombinationsInCond>::const_iterator itVVV;
    int iCond = 0;
    for(itVVV  = m_combinationVector.begin();
            itVVV != m_combinationVector.end(); itVVV++) {

        L1GtLogicParser logicParser(m_algoLogicalExpression, m_algoNumericalExpression);

        std::string condName = logicParser.operandName(iCond);
        bool condResult = logicParser.operandResult(condName);

        myCout << "    Condition " << condName << " evaluated to " << condResult
        << std::endl;


        myCout << "    Object types ";
        myCout << "(";
        ObjectTypeInCond objVec = m_objectTypeVector[iCond];
        for (unsigned int iObj = 0; iObj < objVec.size(); iObj++) {
            L1GtObject obj = objVec[iObj];
            myCout << " " << obj;
        }
        myCout << " ); ";
        myCout << std::endl;


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
        iCond++;
        myCout << "\n\n";
    }
}

