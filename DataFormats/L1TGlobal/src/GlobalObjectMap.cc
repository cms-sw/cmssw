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

// this class header
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"

// system include files
#include <iostream>
#include <iomanip>
#include <iterator>


#include <algorithm>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations




// methods

// return all the combinations passing the requirements imposed in condition condNameVal
const CombinationsInCond* GlobalObjectMap::getCombinationsInCond(
    const std::string& condNameVal) const {

    for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {

        if ((m_operandTokenVector[i]).tokenName == condNameVal) {
            return &(m_combinationVector.at((m_operandTokenVector[i]).tokenNumber));
        }

    }

    // return a null address - should not arrive here
    edm::LogError("GlobalObjectMap")
        << "\n\n  ERROR: The requested condition with tokenName = " << condNameVal
        << "\n  does not exists in the operand token vector."
        << "\n  Returning zero pointer for getCombinationsInCond\n\n" << std::endl;

    return 0;

}

/// return all the combinations passing the requirements imposed in condition condNumberVal
const CombinationsInCond* GlobalObjectMap::getCombinationsInCond(const int condNumberVal) const {

    for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {

        if ((m_operandTokenVector[i]).tokenNumber == condNumberVal) {
            return &(m_combinationVector.at((m_operandTokenVector[i]).tokenNumber));
        }

    }

    // return a null address - should not arrive here
    edm::LogError("GlobalObjectMap")
        << "\n\n  ERROR: The requested condition with tokenNumber = " << condNumberVal
        << "\n  does not exists in the operand token vector."
        << "\n  Returning zero pointer for getCombinationsInCond\n\n" << std::endl;

    return 0;

}
// return the result for the condition condNameVal
const bool GlobalObjectMap::getConditionResult(const std::string& condNameVal) const {

    for (size_t i = 0; i < m_operandTokenVector.size(); ++i) {

        if ((m_operandTokenVector[i]).tokenName == condNameVal) {
            return (m_operandTokenVector[i]).tokenResult;
        }
    }

    // return false - should not arrive here
    edm::LogError("GlobalObjectMap")
        << "\n\n  ERROR: The requested condition with name = " << condNameVal
        << "\n  does not exists in the operand token vector."
        << "\n  Returning false for getConditionResult\n\n" << std::endl;
    return false;

}


void GlobalObjectMap::reset()
{

    // name of the algorithm
    m_algoName.clear();

    // bit number for algorithm
    m_algoBitNumber = -1;

    // GTL result of the algorithm
    m_algoGtlResult = false;

    // vector of operand tokens for an algorithm 
    m_operandTokenVector.clear();
    
    // vector of combinations for all conditions in an algorithm
    m_combinationVector.clear();

}

void GlobalObjectMap::print(std::ostream& myCout) const
{

    myCout << "GlobalObjectMap: print " << std::endl;

    myCout << "  Algorithm name: " << m_algoName << std::endl;
    myCout << "    Bit number: " << m_algoBitNumber << std::endl;
    myCout << "    GTL result: " << m_algoGtlResult << std::endl;

    int operandTokenVectorSize = m_operandTokenVector.size();

    myCout << "    Operand token vector size: " << operandTokenVectorSize;

    if (operandTokenVectorSize == 0) {
        myCout << "   - not properly initialized! " << std::endl;
    }
    else {
        myCout << std::endl;

        for (int i = 0; i < operandTokenVectorSize; ++i) {

            myCout << "      " << std::setw(5) << (m_operandTokenVector[i]).tokenNumber << "\t"
            << std::setw(25) << (m_operandTokenVector[i]).tokenName << "\t" 
            << (m_operandTokenVector[i]).tokenResult 
            << std::endl;
        }

    }

    myCout << "    CombinationVector size: " << m_combinationVector.size() << std::endl;

    myCout << "  conditions: "  << std::endl;

    std::vector<CombinationsInCond>::const_iterator itVVV;
    int iCond = 0;
    for(itVVV  = m_combinationVector.begin();
            itVVV != m_combinationVector.end(); itVVV++) {

        std::string condName = (m_operandTokenVector[iCond]).tokenName;
        bool condResult = (m_operandTokenVector[iCond]).tokenResult;

        myCout << "    Condition " << condName << " evaluated to " << condResult
        << std::endl;

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

