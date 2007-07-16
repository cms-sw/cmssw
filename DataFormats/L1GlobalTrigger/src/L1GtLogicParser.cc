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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

// system include files
#include <string>
#include <vector>
#include <stack>
#include <map>
#include <list>

#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

// constructor(s)

//   from an object map
L1GtLogicParser::L1GtLogicParser(const L1GlobalTriggerObjectMap& objMap)
{

    if ( !setLogicalExpression(objMap.algoLogicalExpression()) ) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = "
        << objMap.algoLogicalExpression()
        << std::endl;

    }

    if ( !setNumericalExpression(objMap.algoNumericalExpression()) ) {
        // error(s) in numerical expression - printed in the relevant place
        throw cms::Exception("FileModule")
        << "\nError in parsing the numerical expression = "
        << objMap.algoNumericalExpression()
        << std::endl;
    }

}

//   from a logical expression
//   numerical expression will be empty
L1GtLogicParser::L1GtLogicParser(const std::string logicalExpressionVal)
{

    // checks also for syntactic correctness of the logical expression

    if ( !setLogicalExpression(logicalExpressionVal) ) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = " << logicalExpressionVal
        << std::endl;

    }

}

//   from a logical and a numerical expression
L1GtLogicParser::L1GtLogicParser(
    const std::string logicalExpressionVal,
    const std::string numericalExpressionVal)
{
    // checks also for correctness

    if ( !setLogicalExpression(logicalExpressionVal) ) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = " << logicalExpressionVal
        << std::endl;

    }

    if ( !setNumericalExpression(numericalExpressionVal) ) {
        // error(s) in numerical expression - printed in the relevant place
        throw cms::Exception("FileModule")
        << "\nError in parsing the numerical expression = " << numericalExpressionVal
        << std::endl;
    }

}

//   from a logical expression,a DecisionWord and a map (string, int)
//   should be used for logical expressions with algorithms
//   the map convert the algorithm name to algorithm bit number, if needed
L1GtLogicParser::L1GtLogicParser(
    const std::string& algoLogicalExpressionVal,
    const DecisionWord& decisionWordVal,
    const std::map<std::string,int>& algoMap)
{

    // checks for correctness

    if ( !setLogicalExpression(algoLogicalExpressionVal) ) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = " << algoLogicalExpressionVal
        << std::endl;

    }

    if ( !setNumericalExpression(decisionWordVal, algoMap) ) {
        // error(s) in numerical expression - printed in the relevant place
        throw cms::Exception("FileModule")
        << "\nError in converting the logical expression = " << algoLogicalExpressionVal
        << " to numerical expression using DecisionWord."
        << std::endl;
    }

}

// destructor
L1GtLogicParser::~L1GtLogicParser()
{
    // empty now
}

// public methods

// return the position index of the operand in the logical expression
int L1GtLogicParser::operandIndex(const std::string operandNameVal) const
{

    int result = -1;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        //LogTrace("L1GtLogicParser")
        //<< "Token string = " << tokenString
        //<< std::endl;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << m_logicalExpression << "'"
            << "\n  Invalid operation/operand " << operandNameVal
            << "\n  Returned index is by default out of range (-1)."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (rpnToken.operand == operandNameVal) {
                result = tmpIndex;

                //LogDebug("L1GtLogicParser")
                //<< "\nL1GtLogicParser::operandIndex - "
                //<< "\nLogical expression = '" << m_logicalExpression << "'"
                //<< "\nIndex of operand " << operandNameVal << " = " << result
                //<< std::endl;

                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "\nLogical expression = '" << m_logicalExpression << "'"
    << "\n  Operand " << operandNameVal << " not found in the logical expression"
    << "\n  Returned index is by default out of range (-1)."
    << std::endl;

    return result;
}

// return the name of the (iOperand)th operand in the logical expression
std::string L1GtLogicParser::operandName(const int iOperand) const
{

    std::string result;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        //LogTrace("L1GtLogicParser")
        //<< "Token string = " << tokenString
        //<< std::endl;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << m_logicalExpression << "'"
            << "\n  Invalid operation/operand at position " << iOperand
            << "\n  Returned empty name by default."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (tmpIndex == iOperand) {
                result = rpnToken.operand;

                //LogDebug("L1GtLogicParser")
                //<< "\nL1GtLogicParser::operandName - "
                //<< "\nLogical expression = '" << m_logicalExpression << "'"
                //<< "\nOperand with index " << iOperand << " = " << result
                //<< std::endl;

                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "\nLogical expression = '" << m_logicalExpression << "'"
    << "\n  No operand found at position " << iOperand
    << "\n  Returned empty name by default."
    << std::endl;

    return result;

}

// return the result for an operand from a logical expression
// using a numerical expression
bool L1GtLogicParser::operandResult(const std::string operandNameVal) const
{

    bool result = false;

    // get the position index of the operand in the logical string
    const int iOperand = operandIndex(operandNameVal);

    result = operandResult(iOperand);

    return result;

}

// return the result for an operand with index iOperand
// in the logical expression using a numerical expression
bool L1GtLogicParser::operandResult(const int iOperand) const
{

    bool result = false;

    // parse the numerical expression

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_numericalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        //LogTrace("L1GtLogicParser")
        //<< "Token string = " << tokenString
        //<< std::endl;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nNumerical expression = '" << m_numericalExpression << "'"
            << "\n  Invalid operation/operand at position " << iOperand
            << "\n  Returned false by default."
            << std::endl;

            result = false;
            return result;
        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (tmpIndex == iOperand) {

                if (rpnToken.operand == "1") {
                    result = true;
                } else {
                    if (rpnToken.operand == "0") {
                        result = false;
                    } else {
                        // something went wrong - break
                        //
                        edm::LogError("L1GtLogicParser")
                        << "\nNumerical expression = '" << m_numericalExpression << "'"
                        << "\n  Invalid result for operand at position " << iOperand
                        << ": " << rpnToken.operand
                        << "\n  It must be 0 or 1"
                        << "\n  Returned false by default."
                        << std::endl;

                        result = false;
                        return result;
                    }
                }

                //LogDebug("L1GtLogicParser")
                //<< "\nL1GtLogicParser::operandResult - "
                //<< "\nNumerical expression = '" << m_numericalExpression << "'"
                //<< "\nResult for operand with index " << iOperand
                //<< " = " << result << "'\n"
                //<< std::endl;

                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "\nNumerical expression = '" << m_numericalExpression << "'"
    << "\n  No operand found at position " << iOperand
    << "\n  Returned false by default."
    << std::endl;

    return result;


}

// return the result for the logical expression
const bool L1GtLogicParser::expressionResult() const
{

    //LogTrace("L1GtLogicParser")
    //<< "\nL1GtLogicParser::expressionResult - "
    //<< std::endl;

    // return false if there is no expression
    if ( m_rpnVector.empty() ) {
        return false;
    }

    // stack containing temporary results
    std::stack<bool> resultStack;
    bool b1, b2;


    for(RpnVector::const_iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {

        //LogTrace("L1GtLogicParser")
        //<< "\nit->operation = " << it->operation
        //<< "\nit->operand =   '" << it->operand << "'\n"
        //<< std::endl;

        switch (it->operation) {

            case OP_OPERAND: {
                    resultStack.push(operandResult(it->operand));
                }

                break;
            case OP_NOT: {
                    b1 = resultStack.top();
                    resultStack.pop();                          // pop the top
                    resultStack.push(!b1);                      // and push the result
                }

                break;
            case OP_OR: {
                    b1 = resultStack.top();
                    resultStack.pop();
                    b2 = resultStack.top();
                    resultStack.pop();
                    resultStack.push(b1 || b2);
                }

                break;
            case OP_AND: {
                    b1 = resultStack.top();
                    resultStack.pop();
                    b2 = resultStack.top();
                    resultStack.pop();
                    resultStack.push(b1 && b2);
                }

                break;
            default: {
                    // should not arrive here
                }

                break;
        }

    }

    // get the result in the top of the stack

    //LogTrace("L1GtLogicParser")
    //<< "\nL1GtLogicParser::expressionResult - "
    //<< "\nLogical expression   = '" << m_logicalExpression << "'"
    //<< "\nNumerical expression = '" << m_numericalExpression << "'"
    //<< "\nResult = " << resultStack.top()
    //<< std::endl;

    return resultStack.top();


}

// convert the logical expression composed with names to
// a logical expression composed with int numbers using
// a (string, int)  map

void L1GtLogicParser::convertNameToIntLogicalExpression(
    const std::map<std::string, int>& nameToIntMap)
{


    if (m_logicalExpression.empty()) {

        return;
    }

    // non-empty logical expression

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    int intValue = -1;

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);
    std::string convertedLogicalExpression;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << m_logicalExpression << "'"
            << "\n  Invalid operation/operand in logical expression."
            << "\n  Return empty logical expression."
            << std::endl;

            m_logicalExpression.clear();
            return;

        }

        if (actualOperation != OP_OPERAND) {

            convertedLogicalExpression.append(getRuleFromType(actualOperation)->opString);

        } else {

            typedef std::map<std::string, int>::const_iterator CIter;

            CIter it = nameToIntMap.find(rpnToken.operand);
            if (it != nameToIntMap.end()) {

                intValue = it->second;
                std::stringstream intStr;
                intStr << intValue;
                convertedLogicalExpression.append(intStr.str());

            } else {

                // it should never be happen
                edm::LogError("L1GtLogicParser")
                << "\nLogical expression = '" << m_logicalExpression << "'"
                << "\n  Could not convert " << rpnToken.operand << " to integer!"
                << "\n  Return empty logical expression."
                << std::endl;

                m_logicalExpression.clear();
                return;
            }

        }

        convertedLogicalExpression.append(" ");   // one whitespace after each token
        lastOperation = actualOperation;
    }

    // remove the last space
    //convertedLogicalExpression.erase(convertedLogicalExpression.size() - 1);
    boost::trim(convertedLogicalExpression);

    LogDebug("L1GtLogicParser")
    << "\nL1GtLogicParser::convertNameToIntLogicalExpression - "
    << "\nLogical expression (strings) = '" << m_logicalExpression << "'"
    << "\nLogical expression (int)     = '" << convertedLogicalExpression << "'\n"
    << std::endl;

    // replace now the logical expression with strings with
    // the converted logical expression

    m_logicalExpression = convertedLogicalExpression;

    return;

}


// return the list of operands for the logical expression
// which are to be used as seeds
const std::list<int> L1GtLogicParser::expressionSeedsOperandList() const
{

    //LogDebug("L1GtLogicParser")
    //<< "\nL1GtLogicParser::expressionSeedsOperandList - "
    //<< "\nLogical expression = '" << m_logicalExpression << "'"
    //<< std::endl;

    // seed list
    std::list<int> opList;

    // temporary results
    std::stack<int> tmpStack;
    std::list<int> tmpList;
    int b1, b2;

    bool newOperandBlock = true;
    bool oneBlockOnly = true;
    bool operandOnly = true;

    for(RpnVector::const_iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {

        //LogTrace("L1GtLogicParser")
        //<< "\nit->operation = " << it->operation
        //<< "\nit->operand =   '" << it->operand << "'\n"
        //<< std::endl;

        switch (it->operation) {

                // RPN always start a block with an operand
            case OP_OPERAND: {

                    // more blocks with operations
                    // push operands from previous block, if any in the tmpList
                    if ( (!newOperandBlock) ) {

                        for (std::list<int>::const_iterator itOp = tmpList.begin();
                                itOp != tmpList.end(); ++itOp) {

                            opList.push_back(*itOp);

                            //LogTrace("L1GtLogicParser")
                            //<< "  Push operand " << *itOp
                            //<<" on the seed operand list"
                            //<< std::endl;

                        }

                        tmpList.clear();

                        newOperandBlock = true;
                        oneBlockOnly = false;

                    }


                    std::istringstream opStream(it->operand);
                    int opInt;

                    if( (opStream >> opInt).fail() ) {

                        edm::LogError("L1GtLogicParser")
                        << "\nLogical expression = '" << m_logicalExpression << "'"
                        << "\n  Conversion to integer failed for " << it->operand
                        << "\n  Returned empty list."
                        << std::endl;

                        opList.clear();
                        return opList;
                    }

                    //LogTrace("L1GtLogicParser")
                    //<< "  Push operand " << opInt << " on the operand stack"
                    //<< std::endl;

                    tmpStack.push(opInt);
                }

                break;
            case OP_NOT: {

                    newOperandBlock = false;
                    operandOnly = false;

                    b1 = tmpStack.top();
                    tmpStack.pop();                          // pop the top

                    tmpStack.push(-1);                       // and push dummy result

                    //LogTrace("L1GtLogicParser")
                    //<< "  Clear tmp operand list"
                    //<< std::endl;

                    tmpList.clear();

                }

                break;
            case OP_OR: {

                    newOperandBlock = false;
                    operandOnly = false;

                    b1 = tmpStack.top();
                    tmpStack.pop();
                    b2 = tmpStack.top();
                    tmpStack.pop();

                    tmpStack.push(-1);                     // and push dummy result

                    if ( b1 >= 0 ) {
                        tmpList.push_back(b1);

                        //LogTrace("L1GtLogicParser")
                        //<< "  Push operand " << b1
                        //<<" on the tmp list"
                        //<< std::endl;
                    }

                    if ( b2 >= 0 ) {
                        tmpList.push_back(b2);

                        //LogTrace("L1GtLogicParser")
                        //<< "  Push operand " << b2
                        //<<" on the tmp list"
                        //<< std::endl;
                    }

                }

                break;
            case OP_AND: {

                    newOperandBlock = false;
                    operandOnly = false;

                    b1 = tmpStack.top();
                    tmpStack.pop();
                    b2 = tmpStack.top();
                    tmpStack.pop();

                    tmpStack.push(-1);                     // and push dummy result


                    if ( b1 >= 0 ) {
                        tmpList.push_back(b1);

                        //LogTrace("L1GtLogicParser")
                        //<< "  Push operand " << b1
                        //<<" on the tmp list"
                        //<< std::endl;
                    }

                    if ( b2 >= 0 ) {
                        tmpList.push_back(b2);

                        //LogTrace("L1GtLogicParser")
                        //<< "  Push operand " << b2
                        //<<" on the tmp list"
                        //<< std::endl;
                    }

                }

                break;
            default: {
                    // should not arrive here
                }

                break;
        }

    }


    // one block only or one operand only
    if ( oneBlockOnly || operandOnly ) {

        // one operand only -
        // there can be only one operand, otherwise one needs an operation
        if (operandOnly) {
            b1 = tmpStack.top();
            tmpList.push_back(b1);
        }

        //
        for (std::list<int>::const_iterator itOp = tmpList.begin();
                itOp != tmpList.end(); ++itOp) {

            opList.push_back(*itOp);

            //LogTrace("L1GtLogicParser")
            //<< "  One block or one operand only: push operand " << *itOp
            //<<" on the seed operand list"
            //<< std::endl;

        }

    }


    return opList;

}

// return the list of indices of the operands in the logical expression
std::list<int> L1GtLogicParser::expressionSeedsOperandIndexList()
{

    //LogDebug("L1GtLogicParser")
    //<< "\nL1GtLogicParser::expressionSeedsOperandIndexList - "
    //<< "\nLogical expression = '" << m_logicalExpression << "'"
    //<< std::endl;


    // build the <operand name, operand index> map

    std::map<std::string, int> operandNameToIndexMap;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    // temporary index for usage in the loop
    int tmpIndex = -1;

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        //LogTrace("L1GtLogicParser")
        //<< "Token string = " << tokenString
        //<< std::endl;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << m_logicalExpression << "'"
            << "\n  Invalid operation/operand " << rpnToken.operand
            << "\n  Returned empty list."
            << std::endl;

            std::list<int> opList;
            opList.clear();
            return opList;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            operandNameToIndexMap[rpnToken.operand] = tmpIndex;

            //LogDebug("L1GtLogicParser")
            //<< "\nL1GtLogicParser::expressionSeedsOperandIndexList - "
            //<< "\noperandNameToIndexMap: name = '" << rpnToken.operand << "'"
            //<< " index = " << tmpIndex
            //<< std::endl;

        }
        lastOperation = actualOperation;
    }

    // convert operand names to indices
    convertNameToIntLogicalExpression(operandNameToIndexMap);
    buildRpnVector(m_logicalExpression);

    // return the list of operands for the logical expression
    return expressionSeedsOperandList();


}



// private methods

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
    OperationType lastOperation, TokenRPN& rpnToken) const
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

/**
 * getRuleFromType Looks for the entry in the operation rules 
 *     and returns a reference if it was found
 *
 * @param oType The type of the operation.
 *
 * @return The reference to the entry or 0 if the Rule was not found.
 *
 */

const L1GtLogicParser::OperationRule* L1GtLogicParser::getRuleFromType(OperationType oType)
{


    int i = 0;

    while (
        (m_operationRules[i].opType != oType) &&
        (m_operationRules[i].opType != OP_NULL) ) {
        i++;
    }

    if (m_operationRules[i].opType == OP_NULL) {
        return 0;
    }

    return &(m_operationRules[i]);
}

/**
 * buildRpnVector Build the postfix notation. 
 *
 * @param expression The expression to be parsed.
 *
 * @return "true" if everything was parsed. "false" if an error occured.
 *
 */

bool L1GtLogicParser::buildRpnVector(const std::string& logicalExpressionVal)
{

    LogDebug("L1GtLogicParser")
    << "\nL1GtLogicParser::buildRpnVector - "
    << "\nLogical expression = '" << logicalExpressionVal << "'\n"
    << std::endl;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    // token as string and as TokenRPN, stack to form the postfix notation
    std::string tokenString;
    TokenRPN rpnToken;
    std::stack<TokenRPN> operatorStack;

    static const std::string whitespaces=" \r\v\n\t";

    // clear possible old rpn vector
    clearRpnVector();

    // stringstream to separate all tokens
    std::istringstream exprStringStream(logicalExpressionVal);

    while ( !exprStringStream.eof() ) {

        exprStringStream >> std::skipws >> std::ws >> tokenString;

        // skip the end
        if (tokenString.find_first_not_of(whitespaces) == std::string::npos ||
                tokenString.length() == 0) {

            LogTrace("L1GtLogicParser")
            << "  Break for token string = " << tokenString
            << std::endl;

            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);

        LogTrace("L1GtLogicParser")
        << "  Token string = '" << tokenString << "'"
        << "\tActual Operation = " << actualOperation
        << std::endl;

        // http://en.wikipedia.org/wiki/Postfix_notation#Converting_from_infix_notation

        switch (actualOperation) {
            case OP_OPERAND: {
                    // operands get pushed to the postfix notation immediately
                    m_rpnVector.push_back(rpnToken);
                }

                break;
            case OP_INVALID: {

                    int errorPosition = exprStringStream.tellg();


                    edm::LogError("L1GtLogicParser")
                    << "\nLogical expression = '" << logicalExpressionVal << "'"
                    << "\n  Syntax error during parsing: "
                    << "\n     " << exprStringStream.str().substr(0,errorPosition)
                    << "\n     " << exprStringStream.str().substr(errorPosition)
                    << "\n  Returned empty RPN vector and result false."
                    << std::endl;

                    // clear the rpn vector before returning
                    clearRpnVector();

                    return false;
                }

                break;
            case OP_NOT: {
                    operatorStack.push(rpnToken);
                    // there are no operators with higher precedence
                }

                break;
            case OP_AND: {
                    // first pop operators with higher precedence (NOT)
                    while (!operatorStack.empty() && operatorStack.top().operation == OP_NOT) {
                        m_rpnVector.push_back(operatorStack.top());
                        operatorStack.pop();
                    }
                    operatorStack.push(rpnToken);
                }

                break;
            case OP_OR: {
                    // pop operators with higher precedence (AND, NOT)
                    while (!operatorStack.empty() &&
                            (operatorStack.top().operation == OP_NOT ||
                             operatorStack.top().operation == OP_AND)  ) {

                        m_rpnVector.push_back(operatorStack.top());
                        operatorStack.pop();
                    }
                    // push operator on stack
                    operatorStack.push(rpnToken);
                }

                break;
            case OP_OPENBRACKET: {

                    // just push it on stack
                    operatorStack.push(rpnToken);
                }

                break;
            case OP_CLOSEBRACKET: {
                    // check if the operatorStack is empty
                    if (operatorStack.empty()) {

                        int errorPosition = exprStringStream.tellg();

                        edm::LogError("L1GtLogicParser")
                        << "\nLogical expression = '" << logicalExpressionVal << "'"
                        << "\n  Syntax error during parsing - misplaced ')':"
                        << "\n     " << exprStringStream.str().substr(0,errorPosition)
                        << "\n     " << exprStringStream.str().substr(errorPosition)
                        << "\n  Returned empty RPN vector and result false."
                        << std::endl;

                        // clear the rpn vector before returning
                        clearRpnVector();

                        return false;
                    }

                    // pop stack until a left parenthesis is found
                    do {
                        if (operatorStack.top().operation != OP_OPENBRACKET) {
                            m_rpnVector.push_back(operatorStack.top()); // pop
                            operatorStack.pop();
                        }
                        if (operatorStack.empty()) { // the operatorStack must not be empty

                            int errorPosition = exprStringStream.tellg();

                            edm::LogError("L1GtLogicParser")
                            << "\nLogical expression = '" << logicalExpressionVal << "'"
                            << "\n  Syntax error during parsing - misplaced ')':"
                            << "\n     " << exprStringStream.str().substr(0,errorPosition)
                            << "\n     " << exprStringStream.str().substr(errorPosition)
                            << "\n  Returned empty RPN vector and result false."
                            << std::endl;

                            // clear the rpn vector before returning
                            clearRpnVector();
                            return false;
                        }
                    } while (operatorStack.top().operation != OP_OPENBRACKET);

                    operatorStack.pop(); // pop the open bracket.
                }

                break;
            default: {
                    // empty
                }
                break;
        }

        lastOperation = actualOperation;    // for the next turn

    }

    // pop the rest of the operator stack
    while (!operatorStack.empty()) {
        if (operatorStack.top().operation == OP_OPENBRACKET) {

            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << logicalExpressionVal << "'"
            << "\n  Syntax error during parsing - missing ')':"
            << "\n  Returned empty RPN vector and result false."
            << std::endl;

            // clear the rpn vector before returning
            clearRpnVector();
            return false;
        }

        m_rpnVector.push_back(operatorStack.top());
        operatorStack.pop();
    }

    // count all operations and check if the result is 1
    int counter = 0;
    for(RpnVector::iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {
        if (it->operation == OP_OPERAND)
            counter++;
        if (it->operation == OP_OR || it->operation == OP_AND)
            counter--;
        if (counter < 1) {

            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << logicalExpressionVal << "'"
            << "\n  Syntax error during parsing - too many operators"
            << "\n  Returned empty RPN vector and result false."
            << std::endl;

            // clear the rpn vector before returning
            clearRpnVector();
            return false;
        }
    }

    if (counter > 1) {

        edm::LogError("L1GtLogicParser")
        << "\nLogical expression = '" << logicalExpressionVal << "'"
        << "\n  Syntax error during parsing - too many operands"
        << "\n  Returned empty RPN vector and result false."
        << std::endl;

        // clear the rpn vector before returning
        clearRpnVector();
        return false;
    }

    return true;
}


// clear rpn vector
void L1GtLogicParser::clearRpnVector()
{

    m_rpnVector.clear();

}


// add spaces before and after parantheses - make separation easier
void L1GtLogicParser::addBracketSpaces(const std::string& srcExpression,
                                       std::string& dstExpression)
{

    static const std::string brackets="()"; // the brackets to be found

    dstExpression = srcExpression;  // copy the string

    size_t position = 0;
    while ( (position = dstExpression.find_first_of(brackets, position)) != std::string::npos ) {

        // add space after if none is there
        if (dstExpression[position + 1] != ' ') {
            dstExpression.insert(position + 1, " ");
        }

        // add space before if none is there
        if (dstExpression[position - 1] != ' ') {
            dstExpression.insert(position, " ");
            position++;
        }
        position++;
    }
}


// set the logical expression - check for correctness the input string
bool L1GtLogicParser::setLogicalExpression(const std::string& logicalExpressionVal)
{

    // add spaces around brackets
    std::string logicalExpressionBS;
    addBracketSpaces(logicalExpressionVal, logicalExpressionBS);

    // trim leading or trailing spaces
    boost::trim(logicalExpressionBS);

    clearRpnVector();

    if ( !buildRpnVector(logicalExpressionBS) ) {
        m_logicalExpression = "";
        return false;
    }

    m_logicalExpression = logicalExpressionBS;

    LogDebug("L1GtLogicParser")
    << "\nL1GtLogicParser::setLogicalExpression - "
    << "\nLogical expression = '" << m_logicalExpression << "'\n"
    << std::endl;

    return true;

}

// set the numerical expression (the logical expression with each operand
// replaced with the value) from a string
// check also for correctness the input string
bool L1GtLogicParser::setNumericalExpression(const std::string& numericalExpressionVal)
{

    // add spaces around brackets
    std::string numericalExpressionBS;
    addBracketSpaces(numericalExpressionVal, numericalExpressionBS);

    // check for consistency with the logical expression
    // TODO FIXME

    // trim leading or trailing spaces
    boost::trim(numericalExpressionBS);

    m_numericalExpression = numericalExpressionBS;

    LogDebug("L1GtLogicParser")
    << "\nL1GtLogicParser::setNumericalExpression - "
    << "\nNumerical Expression = '" << m_numericalExpression << "'\n"
    << std::endl;

    return true;

}


// convert the logical expression composed with algorithm bits/names into a
// numerical expression using values from DecisionWord.
// the map convert from algorithm name to algorithm bit number, if needed

bool L1GtLogicParser::setNumericalExpression(const DecisionWord& decisionWordVal,
        const std::map<std::string, int>& algoMap)
{


    if (m_logicalExpression.empty()) {

        m_numericalExpression.clear();
        return true;
    }

    // non-empty logical expression

    m_numericalExpression.clear();

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "\nLogical expression = '" << m_logicalExpression << "'"
            << "\n  Invalid operation/operand in logical expression."
            << "\n  Return empty numerical expression."
            << std::endl;

            m_numericalExpression.clear();

            return false;

        }

        if (actualOperation != OP_OPERAND) {

            m_numericalExpression.append(getRuleFromType(actualOperation)->opString);

        } else {

            // replace the operand with its result
            if (operandResultDecisionWord(rpnToken.operand, decisionWordVal, algoMap)) {
                m_numericalExpression.append("1"); // true
            } else {
                m_numericalExpression.append("0"); // false
            }
        }

        m_numericalExpression.append(" ");         // one whitespace after each token
        lastOperation = actualOperation;
    }

    // remove leading and trailing spaces
    //m_numericalExpression.erase(m_numericalExpression.size() - 1);
    boost::trim(m_numericalExpression);

    LogDebug("L1GtLogicParser")
    << "\nL1GtLogicParser::setNumericalExpression - "
    << "\nNumerical Expression = '" << m_numericalExpression << "'\n"
    << std::endl;

    return true;

}


// return the result for an operand from a logical expression
// from a decision word
bool L1GtLogicParser::operandResultDecisionWord(
    const std::string& operandIdentifier,
    const DecisionWord& decisionWordVal,
    const std::map<std::string, int>& algoMap)
{
    bool resultOperand = false;

    // convert string to integer
    std::istringstream opStream(operandIdentifier);
    unsigned int opBitNumber;

    bool flagName = (opStream >> opBitNumber).fail();

    if ( flagName) {

        // the logical expression contains L1 trigger names, get
        // the corresponding bit number - expensive operation
        std::map<std::string, int>::const_iterator itMap = algoMap.find(operandIdentifier);

        if (itMap != algoMap.end()) {
            opBitNumber = itMap->second;
        } else {

            // it should always find a bit number - throw exception?
            edm::LogError("L1GtLogicParser")
            << "\nBit number not found for operand" << operandIdentifier
            << "\n  No such entry in the L1 Trigger menu."
            << "\n  Returned false by default."
            << std::endl;

            return resultOperand;

        }

    }

    // acces bit

    const unsigned int numberTriggerBits = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {

        if (opBitNumber == iBit) {
            resultOperand = decisionWordVal[iBit];

            return resultOperand;
        }

    }

    // it should always find a bit number
    edm::LogError("L1GtLogicParser")
    << "\nBit number not found for operand" << operandIdentifier
    << " converted to " << opBitNumber
    << "\n  No such entry in the L1 Trigger menu."
    << "\n  Returned false by default."
    << std::endl;

    return resultOperand;

}

// static members

// rules for operations
// 1st column: operation string
// 2nd column: operation type
// 3rd column: forbiddenLastOperation (what operation the operator/operand must not follow)
const struct L1GtLogicParser::OperationRule L1GtLogicParser::m_operationRules[] =
    {
        { "AND",  OP_AND,           OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
        { "OR",   OP_OR,            OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
        { "NOT",  OP_NOT,           OP_OPERAND | OP_CLOSEBRACKET                       },
        { "(",    OP_OPENBRACKET,   OP_OPERAND | OP_CLOSEBRACKET                       },
        { ")",    OP_CLOSEBRACKET,  OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET           },
        { NULL,   OP_OPERAND,       OP_OPERAND | OP_CLOSEBRACKET                       },
        { NULL,   OP_NULL,          OP_NULL                                            }
    };
