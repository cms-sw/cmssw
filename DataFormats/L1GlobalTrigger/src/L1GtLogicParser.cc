/**
 * \class L1GtLogicParser
 * 
 * 
 * Description: see header file.  
 *
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
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

// system include files
#include <string>
#include <vector>

#include <iostream>
#include <sstream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// constructor(s)

//   from an object map
L1GtLogicParser::L1GtLogicParser(const L1GlobalTriggerObjectMap& objMap)
{
    m_algoLogicalExpression = objMap.algoLogicalExpression();
    m_algoNumericalExpression = objMap.algoNumericalExpression();

}

///   from a logical and a numerical expression
L1GtLogicParser::L1GtLogicParser(const std::string algoLogicalExpressionVal,
                                 const std::string algoNumericalExpressionVal)
{
    m_algoLogicalExpression = algoLogicalExpressionVal;
    m_algoNumericalExpression = algoNumericalExpressionVal;
}

// destructor
L1GtLogicParser::~L1GtLogicParser()
{}

// methods

/**
 * getOperation Get the operation from a string and check if it is allowed
 *
 * @param tokenString   The string to examine.
 * @param lastOperation The last operation.
 * @param rpnToken      The destination where the token for postfix notation is written to.
 *
 * @return              The Operation type or OP_INVALID, if the operation is not allowed
 *
 */

L1GtLogicParser::OperationType L1GtLogicParser::getOperation(
    const std::string& tokenString,
    OperationType lastOperation, TokenRPN& rpnToken)
{

    OperationType actualOperation = OP_OPERAND;    // default value

    int i = 0;

    while (m_operationRules[i].opType != OP_OPERAND) {
        if (tokenString == m_operationRules[i].opString) {
            actualOperation = (OperationType) m_operationRules[i].opType;
            break;
        }
        i++;
    }

    // check if the operation is allowed
    if (m_operationRules[i].forbiddenLastOperation & lastOperation) {
        return OP_INVALID;
    }

    //
    if (actualOperation == OP_OPERAND) {

        rpnToken.operand = tokenString;

    } else {

        rpnToken.operand = "";
    }

    rpnToken.operation = actualOperation;

    // else we got a valid operation
    return actualOperation;
}



// return the index of the condition in the logical (numerical) expression
// this index can then be associated with the index of the CombinationsInCond
// in std::vector<CombinationsInCond> of that object map
int L1GtLogicParser::conditionIndex(const std::string condNameVal)
{

    int result = -1;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_algoLogicalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand for condition" << condNameVal
            << " \n  Returned index is by default out of range (-1)."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (rpnToken.operand == condNameVal) {
                result = tmpIndex;
                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  Condition " << condNameVal << " not found in algorithm logical expression: "
    << m_algoLogicalExpression
    << " \n  Returned index is by default out of range (-1)."
    << std::endl;

    return result;
}

/// return the name of the (iCondition)th condition in the algorithm logical expression
std::string L1GtLogicParser::conditionName(const int iCondition)
{

    std::string result;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_algoLogicalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand for condition at position " << iCondition
            << " \n  Returned empty name by default."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (tmpIndex == iCondition) {
                result = rpnToken.operand;
                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  No condition at position " << iCondition 
    << " found in algorithm logical expression: "
    << m_algoLogicalExpression
    << " \n  Returned empty name by default."
    << std::endl;

    return result;

}

/// return the result for a condition in an algorithm
bool L1GtLogicParser::conditionResult(const std::string condNameVal)
{

    bool result = false;

    // get the index of the condition in the logical string
    int iCondition = conditionIndex(condNameVal);

    // parse the numerical expression

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_algoNumericalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand for condition" << condNameVal
            << " \n  Returned result is by default false."
            << std::endl;

            return result;
        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (tmpIndex == iCondition) {

                if (rpnToken.operand == "1") {
                    result = true;
                } else {
                    if (rpnToken.operand == "0") {
                        result = false;
                    } else {
                        // something went wrong - break
                        //
                        edm::LogError("L1GtLogicParser")
                        << "  Result for condition " << condNameVal << " is "
                        << rpnToken.operand << "; it must be 0 or 1."
                        << " \n  Returned result is set by default to false."
                        << std::endl;
                    }
                }

                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  Condition " << condNameVal << " not found in algorithm logical expression: "
    << m_algoLogicalExpression
    << " \n  Returned result is set by default to false."
    << std::endl;

    return result;

}


// rules for operations
// 1st column: operation string
// 2nd column: operation type
// 3rd column: forbiddenLastOperation (what operation the operator/operand must not follow)
const struct L1GtLogicParser::OperationRule
            L1GtLogicParser::m_operationRules[] =
    {

        { "AND", OP_AND,          OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL
        },
        { "OR",  OP_OR,           OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
        { "NOT", OP_NOT,          OP_OPERAND | OP_CLOSEBRACKET | OP_NULL},
        { "(",   OP_OPENBRACKET,  OP_OPERAND | OP_CLOSEBRACKET },
        { ")",   OP_CLOSEBRACKET, OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET },
        { NULL,  OP_OPERAND,      OP_OPERAND | OP_CLOSEBRACKET },           // default
        { NULL,  OP_NULL,         OP_NULL }
    };
