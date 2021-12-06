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
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"

// system include files
#include <string>

#include <queue>
#include <stack>
#include <vector>

#include <iomanip>
#include <iostream>

#include <boost/algorithm/string.hpp>

// user include files

//
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

/// constructor from an algorithm from event setup
L1GtAlgorithmEvaluation::L1GtAlgorithmEvaluation(const L1GtAlgorithm &alg)
    : m_algoResult(false), m_logicalExpression(alg.algoLogicalExpression()), m_rpnVector(alg.algoRpnVector()) {
  // the rest is properly initialized by default
}

// methods

/// evaluate an algorithm
void L1GtAlgorithmEvaluation::evaluateAlgorithm(const int chipNumber,
                                                const std::vector<ConditionEvaluationMap> &conditionResultMaps) {
  // set result to false if there is no expression
  if (m_rpnVector.empty()) {
    m_algoResult = false;

    // it should never be happen
    throw cms::Exception("FailModule") << "\nEmpty RPN vector for the logical expression = " << m_logicalExpression
                                       << std::endl;
  }

  // reserve memory
  int rpnVectorSize = m_rpnVector.size();

  m_algoCombinationVector.reserve(rpnVectorSize);
  m_operandTokenVector.reserve(rpnVectorSize);

  // stack containing temporary results
  // FIXME we shall find a better solution than static
  std::stack<bool, std::vector<bool>> resultStack;
  bool b1, b2;

  int opNumber = 0;

  for (RpnVector::const_iterator it = m_rpnVector.begin(); it != m_rpnVector.end(); it++) {
    // LogTrace("L1GlobalTrigger")
    //<< "\nit->operation = " << it->operation
    //<< "\nit->operand =   '" << it->operand << "'\n"
    //<< std::endl;

    switch (it->operation) {
      case L1GtLogicParser::OP_OPERAND: {
        CItEvalMap itCond = (conditionResultMaps.at(chipNumber)).find(it->operand);
        if (itCond != (conditionResultMaps[chipNumber]).end()) {
          if (nullptr == itCond->second) {
            // it should never be happen, only valid conditions are in the maps
            throw cms::Exception("FailModule")
                << "\nCondition " << (it->operand) << " NULL pointer found in condition map" << std::endl;
          }

          //
          bool condResult = (itCond->second)->condLastResult();

          resultStack.push(condResult);

          // only conditions are added to /counted in m_operandTokenVector
          // opNumber is the index of the condition in the logical expression
          OperandToken opToken;
          opToken.tokenName = it->operand;
          opToken.tokenNumber = opNumber;
          opToken.tokenResult = condResult;

          m_operandTokenVector.push_back(opToken);
          opNumber++;

          //
          CombinationsInCond const &combInCondition = (itCond->second)->getCombinationsInCond();
          m_algoCombinationVector.push_back(combInCondition);

        } else {
          // it should never be happen, all conditions are in the maps
          throw cms::Exception("FailModule")
              << "\nCondition " << (it->operand) << " not found in condition map" << std::endl;
        }

      }

      break;
      case L1GtLogicParser::OP_NOT: {
        b1 = resultStack.top();
        resultStack.pop();      // pop the top
        resultStack.push(!b1);  // and push the result
      }

      break;
      case L1GtLogicParser::OP_OR: {
        b1 = resultStack.top();
        resultStack.pop();
        b2 = resultStack.top();
        resultStack.pop();
        resultStack.push(b1 || b2);
      }

      break;
      case L1GtLogicParser::OP_AND: {
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

  // clear resultStack not needed since it is now a function temporary
}

// print algorithm evaluation
void L1GtAlgorithmEvaluation::print(std::ostream &myCout) const {
  myCout << std::endl;

  myCout << "    Algorithm result:          " << m_algoResult << std::endl;

  myCout << "    CombinationVector size:    " << m_algoCombinationVector.size() << std::endl;

  int operandTokenVectorSize = m_operandTokenVector.size();

  myCout << "    Operand token vector size: " << operandTokenVectorSize;

  if (operandTokenVectorSize == 0) {
    myCout << "   - not properly initialized! " << std::endl;
  } else {
    myCout << std::endl;

    for (int i = 0; i < operandTokenVectorSize; ++i) {
      myCout << "      " << std::setw(5) << (m_operandTokenVector[i]).tokenNumber << "\t" << std::setw(25)
             << (m_operandTokenVector[i]).tokenName << "\t" << (m_operandTokenVector[i]).tokenResult << std::endl;
    }
  }

  myCout << std::endl;
}
