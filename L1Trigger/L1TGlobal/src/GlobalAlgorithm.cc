/**
 * \class GlobalAlgorithm
 *
 *
 * Description: L1 GT algorithm.
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
#include "L1Trigger/L1TGlobal/interface/GlobalAlgorithm.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

// forward declarations

// constructor(s)
//   default
GlobalAlgorithm::GlobalAlgorithm()
{
    // default values for private members not set
    // the other private members are C++ initialized
    m_algoBitNumber = -1;
    m_algoChipNumber = -1;

}

//   name only
GlobalAlgorithm::GlobalAlgorithm(const std::string& algoNameValue) :
    m_algoName(algoNameValue) {

    // default values for private members not set
    // the other private members are C++ initialized
    m_algoBitNumber = -1;
    m_algoChipNumber = -1;

}

//   name and logical expression
GlobalAlgorithm::GlobalAlgorithm(
        const std::string& algoNameValue, const std::string& algoLogicalExpressionValue) :
    m_algoName(algoNameValue), m_algoLogicalExpression(algoLogicalExpressionValue) {

    GlobalLogicParser logicParser(m_algoLogicalExpression);
    m_algoRpnVector = logicParser.rpnVector();

    // default values for private members not set
    m_algoBitNumber = -1;
    m_algoChipNumber = -1;
}

//   name, logical expression and bit number
GlobalAlgorithm::GlobalAlgorithm(
        const std::string& algoNameValue, const std::string& algoLogicalExpressionValue,
        const int algoBitNumberValue) :
    m_algoName(algoNameValue), m_algoLogicalExpression(algoLogicalExpressionValue),
            m_algoBitNumber(algoBitNumberValue)

{
    GlobalLogicParser logicParser(m_algoLogicalExpression);
    m_algoRpnVector = logicParser.rpnVector();

    // default values for private members not set
    m_algoChipNumber = -1;

}

// destructor
GlobalAlgorithm::~GlobalAlgorithm()
{
    // empty
}

// public methods

// get the condition chip number the algorithm is located on
const int GlobalAlgorithm::algoChipNumber(const int numberConditionChips,
                                    const int pinsOnConditionChip,
                                    const std::vector<int>& orderConditionChip) const
{
    int posChip = (m_algoBitNumber/pinsOnConditionChip) + 1;
    for (int iChip = 0; iChip < numberConditionChips; ++iChip) {
        if (posChip == orderConditionChip[iChip]) {
            return iChip;
        }
    }

    // chip number not found
    return -1;
}

// get the output pin on the condition chip for the algorithm
const int GlobalAlgorithm::algoOutputPin(const int numberConditionChips,
                                       const int pinsOnConditionChip,
                                       const std::vector<int>& orderConditionChip) const
{

    int iChip = algoChipNumber(numberConditionChips, pinsOnConditionChip, orderConditionChip);

    int outputPin = m_algoBitNumber - (orderConditionChip[iChip] -1)*pinsOnConditionChip + 1;

    return outputPin;
}



// print algorithm
void GlobalAlgorithm::print(std::ostream& myCout) const {

    myCout << std::endl;

    myCout << "    Algorithm name:         " << m_algoName << std::endl;
    myCout << "    Algorithm alias:        " << m_algoAlias << std::endl;

    myCout << "    Bit number:             " << m_algoBitNumber;
    if (m_algoBitNumber < 0) {
        myCout << "   - not properly initialized! " << std::endl;
    }
    else {
        myCout << std::endl;
    }

    myCout << "    Located on chip number: " << m_algoChipNumber;
    if (m_algoChipNumber < 0) {
        myCout << "   - not properly initialized! " << std::endl;
    }
    else {
        myCout << std::endl;
    }

    myCout << "    Logical expresssion:    " << m_algoLogicalExpression << std::endl;

    int rpnVectorSize = m_algoRpnVector.size();

    myCout << "    RPN vector size:        " << rpnVectorSize;

    if (rpnVectorSize == 0) {
        myCout << "   - not properly initialized! " << std::endl;
    }
    else {
        myCout << std::endl;

        for (int i = 0; i < rpnVectorSize; ++i) {

            myCout << "      ( " << (m_algoRpnVector[i]).operation << ", "
            << (m_algoRpnVector[i]).operand << " )" << std::endl;
        }

    }

    myCout << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const GlobalAlgorithm& result)
{
    result.print(os);
    return os;

}
