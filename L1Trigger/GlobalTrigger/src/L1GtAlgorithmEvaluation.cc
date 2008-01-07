/**
 * \class L1GtAlgorithmEvaluation
 * 
 * 
 * Description: Evaluation of a L1 Global Trigger algorithm.
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
#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"

// system include files
#include <string>

#include <stack>
#include <queue>
#include <vector>

#include <iostream>

#include <boost/algorithm/string.hpp>

// user include files

//   base class
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// constructor
L1GtAlgorithmEvaluation::L1GtAlgorithmEvaluation() :
    L1GtLogicParser() {

    m_algoResult = false;

    // the rest is properly initialized by default
}

/// constructor from an algorithm from event setup
L1GtAlgorithmEvaluation::L1GtAlgorithmEvaluation(const L1GtAlgorithm* alg) :
    L1GtLogicParser(alg->algoLogicalExpression()) {

    m_algoResult = false;

    // the rest is properly initialized by default

}

// copy constructor
L1GtAlgorithmEvaluation::L1GtAlgorithmEvaluation(L1GtAlgorithmEvaluation& cp) {

    // parser part
    m_logicalExpression = cp.logicalExpression();
    m_numericalExpression = cp.numericalExpression();
    RpnVector m_rpnVector = cp.rpnVector();

    // L1GtAlgorithmEvaluation part
    m_algoResult = cp.gtAlgoResult();
    m_algoNumericalExpression = cp.gtAlgoNumericalExpression();
    m_algoCombinationVector = *(cp.gtAlgoCombinationVector());

}

// destructor
L1GtAlgorithmEvaluation::~L1GtAlgorithmEvaluation() {

    // empty

}

// methods

/// evaluate an algorithm
void L1GtAlgorithmEvaluation::evaluateAlgorithm(const int chipNumber,
    const std::vector<std::map<std::string, L1GtConditionEvaluation*> >& conditionResultMaps) {

    // set result to false if there is no expression 
    if (m_rpnVector.empty() ) {
        m_algoResult = false;

        // it should never be happen
        throw cms::Exception("FailModule")
        << "\nEmpty RPN vector for the logical expression = "
        << m_logicalExpression
        << std::endl;

    }

    // stack containing temporary results
    std::stack<bool> resultStack;
    bool b1, b2;

    // stack of condition results - to be used for numerical expression
    std::queue<int> condResultStack;

    typedef std::map<std::string, L1GtConditionEvaluation*>::const_iterator CondIter;

    for (RpnVector::const_iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {

        //LogTrace("L1GtAlgorithmEvaluation")
        //<< "\nit->operation = " << it->operation
        //<< "\nit->operand =   '" << it->operand << "'\n"
        //<< std::endl;

        switch (it->operation) {

            case OP_OPERAND: {

                CondIter itCond = (conditionResultMaps.at(chipNumber)).find(it->operand);
                if (itCond != (conditionResultMaps[chipNumber]).end()) {

                    //
                    bool condResult = (itCond->second)->condLastResult();

                    resultStack.push(condResult);
                    condResultStack.push(condResult);

                    //
                    CombinationsInCond* combInCondition = (itCond->second)->getCombinationsInCond();
                    m_algoCombinationVector.push_back(*combInCondition);

                }
                else {

                    // it should never be happen, all conditions are in the maps
                    throw cms::Exception("FailModule")
                    << "\nCondition " << (itCond->first) << "not found in condition map"
                    << std::endl;

                }

            }

                break;
            case OP_NOT: {
                b1 = resultStack.top();
                resultStack.pop(); // pop the top
                resultStack.push(!b1); // and push the result
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

    m_algoResult = resultStack.top();

    // convert the logical expression to the numerical expression using the saved stack 

    m_algoNumericalExpression.clear();

    OperationType actualOperation = OP_NULL;
    OperationType lastOperation = OP_NULL;

    std::string tokenString;
    TokenRPN rpnToken; // token to be used by getOperation

    // stringstream to separate all tokens
    std::istringstream exprStringStream(m_logicalExpression);

    while (!exprStringStream.eof()) {

        exprStringStream >> tokenString;

        actualOperation = getOperation(tokenString, lastOperation, rpnToken);
        if (actualOperation == OP_INVALID) {

            m_algoNumericalExpression.clear();

            // it should never be invalid
            throw cms::Exception("FailModule")
            << "\nLogical expression = '"
            << m_logicalExpression << "'"
            << "\n  Invalid operation/operand in logical expression."
            << std::endl;

        }

        if (actualOperation != OP_OPERAND) {

            m_algoNumericalExpression.append(getRuleFromType(actualOperation)->opString);

        }
        else {

            // replace the operand with its result
            if (condResultStack.front()) {
                m_algoNumericalExpression.append("1"); // true
            }
            else {
                m_algoNumericalExpression.append("0"); // false
            }

            condResultStack.pop();
        }

        m_algoNumericalExpression.append(" "); // one whitespace after each token
        lastOperation = actualOperation;
    }

    // remove leading and trailing spaces
    boost::trim(m_algoNumericalExpression);

    //LogTrace("L1GtAlgorithmEvaluation")
    //<< "\nLogical expression   = '" << m_logicalExpression << "'"
    //<< "\nNumerical expression = '" << m_algoNumericalExpression << "'"
    //<< "\nResult = " << m_algoResult
    //<< std::endl;


}

// print algorithm evaluation
void L1GtAlgorithmEvaluation::print(std::ostream& myCout) const {

    myCout << std::endl;

    myCout << "    Algorithm result:       " << m_algoResult << std::endl;
    myCout << "    Numerical expression:   '" << m_algoNumericalExpression << "'" << std::endl;

    myCout << "    CombinationVector size: " << m_algoCombinationVector.size() << std::endl;

    myCout << std::endl;
}

