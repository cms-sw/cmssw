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
    // both expressions are checked for correctness before they are set in
    // L1GlobalTriggerObjectMap - no checks here
    m_logicalExpression = objMap.algoLogicalExpression();
    m_numericalExpression = objMap.algoNumericalExpression();

}

//   from a logical and a numerical expression
L1GtLogicParser::L1GtLogicParser(
    const std::string logicalExpressionVal,
    const std::string numericalExpressionVal)
{
    // checks also for correctness

    if (setLogicalExpression(logicalExpressionVal)) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = " << logicalExpressionVal
        << std::endl;

    }

    if (setNumericalExpression(numericalExpressionVal)) {
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

    if (setLogicalExpression(algoLogicalExpressionVal)) {

        // error(s) in logical expression - printed in the relevant place
        throw cms::Exception("FailModule")
        << "\nError in parsing the logical expression = " << algoLogicalExpressionVal
        << std::endl;

    }

    if (setNumericalExpression(decisionWordVal, algoMap)) {
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
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand " << operandNameVal
            << " \n  Returned index is by default out of range (-1)."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (rpnToken.operand == operandNameVal) {
                result = tmpIndex;
                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  Operand " << operandNameVal << " not found in the logical expression: "
    << m_logicalExpression
    << " \n  Returned index is by default out of range (-1)."
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
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand at position " << iOperand
            << " \n  Returned empty name by default."
            << std::endl;

            return result;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            if (tmpIndex == iOperand) {
                result = rpnToken.operand;
                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  No operand at position " << iOperand
    << " found in the logical expression: "
    << m_logicalExpression
    << " \n  Returned empty name by default."
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
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand " << iOperand
            << " \n  Returned result is by default false."
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
                        << "  Result for operand " << iOperand << " is "
                        << rpnToken.operand << "; it must be 0 or 1."
                        << " \n  Returned result is set by default to false."
                        << std::endl;

                        result = false;
                        return result;
                    }
                }

                return result;
            }
        }
        lastOperation = actualOperation;
    }

    //
    edm::LogError("L1GtLogicParser")
    << "  Operand " << iOperand << " not found in the logical expression: "
    << m_logicalExpression
    << " \n  Returned result is set by default to false."
    << std::endl;

    return result;


}

// return the result for the logical expression
const bool L1GtLogicParser::expressionResult() const
{

    // return false if there is no expression
    if ( m_rpnVector.empty() ) {
        return false;
    }

    // stack containing temporary results
    std::stack<bool> resultStack;
    bool b1, b2;


    for(RpnVector::const_iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {

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
    return resultStack.top();


}

// return the list of operands for the logical expression
// which are to be used as seeds
// TODO NOT treatment
const std::list<int> L1GtLogicParser::expressionOperandList() const
{

    std::list<int> opList;

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation   = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken;           // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand " << rpnToken.operand
            << " \n  Returned empty list."
            << std::endl;

            opList.clear();
            return opList;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {
            
            // FIXME HERE  
            // convert string to integer first

            std::istringstream opStream(rpnToken.operand);
            int opInt;

            if( (opStream >> opInt).fail() ) {
                edm::LogError("L1GtLogicParser")
                << "  Conversion to integer failed for " << rpnToken.operand
                << " \n  Returned empty list."
                << std::endl;

                opList.clear();
                return opList;
            }

            opList.push_back(opInt);

        }
        lastOperation = actualOperation;
    }

    return opList;



}

// return the list of indices of the operands in the logical expression
const std::list<int> L1GtLogicParser::expressionOperandIndexList() const
{

    std::list<int> opList;

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
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand " << rpnToken.operand
            << " \n  Returned empty list."
            << std::endl;

            opList.clear();
            return opList;

        }

        if (actualOperation != OP_OPERAND) {

            // do nothing

        } else {

            tmpIndex++;
            opList.push_back(tmpIndex);

        }
        lastOperation = actualOperation;
    }

    return opList;


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
 * @return 0 if everything was parsed. -1 if an error occured.
 *
 */

int L1GtLogicParser::buildRpnVector(const std::string& logicalExpressionVal)
{

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
                tokenString.length() == 0 ||
                exprStringStream.eof()) {

            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);

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
                    << "    Syntax error while parsing expression: "
                    <<  logicalExpressionVal << "\n"
                    << "    " << exprStringStream.str().substr(0,errorPosition)
                    << " <-- ERROR!" << "\n"
                    << exprStringStream.str().substr(errorPosition)
                    << std::endl;

                    // clear the rpn vector before returning
                    clearRpnVector();

                    return -1;
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
                        << "    Syntax error while parsing expresssion - misplaced ')': "
                        << logicalExpressionVal << "\n"
                        << exprStringStream.str().substr(0,errorPosition)
                        << "<-- ERROR!" << "\n"
                        << exprStringStream.str().substr(errorPosition)
                        << std::endl;

                        // clear the rpn vector before returning
                        clearRpnVector();

                        return -1;
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
                            << "    Syntax error while parsing expresssion - misplaced ')': "
                            << logicalExpressionVal << "\n"
                            << exprStringStream.str().substr(0,errorPosition)
                            << "<-- ERROR!" << "\n"
                            << exprStringStream.str().substr(errorPosition)
                            << std::endl;

                            // clear the rpn vector before returning
                            clearRpnVector();
                            return -1;
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
            << "    Syntax error while parsing expression - missing ')': "
            << logicalExpressionVal
            << std::endl;

            // clear the rpn vector before returning
            clearRpnVector();
            return -1;
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
            << "    Syntax error while parsing expression (too many operators) : "
            << logicalExpressionVal
            << std::endl;

            // clear the rpn vector before returning
            clearRpnVector();
            return -1;
        }
    }

    if (counter > 1) {
        edm::LogError("L1GtLogicParser")
        << "    Syntax error while parsing algorithm (too many operands) : "
        << logicalExpressionVal
        << std::endl;

        // clear the rpn vector before returning
        clearRpnVector();
        return -1;
    }

    return 0;
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
int L1GtLogicParser::setLogicalExpression(const std::string& logicalExpressionVal)
{

    // add spaces around brackets
    std::string logicalExpressionBS;
    addBracketSpaces(logicalExpressionVal, logicalExpressionBS);


    clearRpnVector();

    if (buildRpnVector(logicalExpressionBS) != 0) {
        m_logicalExpression = "";
        return -1;
    }

    m_logicalExpression = logicalExpressionBS;

    return 0;

}

// set the numerical expression (the logical expression with each operand
// replaced with the value) from a string
// check also for correctness the input string
int L1GtLogicParser::setNumericalExpression(const std::string& numericalExpressionVal)
{

    // add spaces around brackets
    std::string numericalExpressionBS;
    addBracketSpaces(numericalExpressionVal, numericalExpressionBS);

    // check for consistency with the logical expression
    // TODO FIXME

    m_numericalExpression = numericalExpressionBS;

    return 0;

}


// convert the logical expression composed with algorithm bits/names into a
// numerical expression using values from DecisionWord.
// the map convert from algorithm name to algorithm bit number, if needed

int L1GtLogicParser::setNumericalExpression(const DecisionWord& decisionWordVal,
        const std::map<std::string, int>& algoMap)
{


    if (m_logicalExpression.empty()) {

        m_numericalExpression.clear();
        return 0;
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
        if (exprStringStream.eof()) {
            break;
        }

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            // it should never be invalid
            edm::LogError("L1GtLogicParser")
            << "  Invalid operation/operand in logical expression" << m_logicalExpression
            << std::endl;

            m_numericalExpression.clear();

            return -1;

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

        lastOperation = actualOperation;
    }

    return 0;

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
            << "  Bit number not found for operand" << operandIdentifier
            << "\n  No such entry in the L1 Trigger menu."
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
    << "  Bit number not found for operand" << operandIdentifier
    << " converted to " << opBitNumber
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
        { "AND",  OP_AND,           OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL
        },
        { "OR",   OP_OR,            OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
        { "NOT",  OP_NOT,           OP_OPERAND | OP_CLOSEBRACKET | OP_NULL             },
        { "(",    OP_OPENBRACKET,   OP_OPERAND | OP_CLOSEBRACKET                       },
        { ")",    OP_CLOSEBRACKET,  OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET           },
        { NULL,   OP_OPERAND,       OP_OPERAND | OP_CLOSEBRACKET                       },
        { NULL,   OP_NULL,          OP_NULL                                            }
    };
