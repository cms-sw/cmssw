#ifndef GlobalTrigger_L1GtAlgorithmEvaluation_h
#define GlobalTrigger_L1GtAlgorithmEvaluation_h

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

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

// forward declarations
class L1GtAlgorithm;
class L1GtConditionEvaluation;

// class interface
class L1GtAlgorithmEvaluation {
public:
  typedef L1GtLogicParser::TokenRPN TokenRPN;
  typedef std::vector<TokenRPN> RpnVector;
  typedef L1GtLogicParser::OperandToken OperandToken;

  /// constructor
  //  L1GtAlgorithmEvaluation();

  /// constructor from an algorithm from event setup
  explicit L1GtAlgorithmEvaluation(const L1GtAlgorithm &);

  /// copy constructor
  // L1GtAlgorithmEvaluation(L1GtAlgorithmEvaluation&);

  /// destructor
  // virtual ~L1GtAlgorithmEvaluation();

  // typedef std::map<std::string, L1GtConditionEvaluation*>
  // ConditionEvaluationMap;
  typedef std ::unordered_map<std::string, L1GtConditionEvaluation *> ConditionEvaluationMap;
  typedef ConditionEvaluationMap::const_iterator CItEvalMap;
  typedef ConditionEvaluationMap::iterator ItEvalMap;

public:
  /// get / set the result of the algorithm
  inline bool gtAlgoResult() const { return m_algoResult; }

  inline void setGtAlgoResult(const bool algoResult) { m_algoResult = algoResult; }

  /// evaluate an algorithm
  void evaluateAlgorithm(const int chipNumber, const std::vector<ConditionEvaluationMap> &);

  /// get all the object combinations evaluated to true in the conditions
  /// from the algorithm
  inline std::vector<CombinationsInCond> &gtAlgoCombinationVector() { return m_algoCombinationVector; }

  inline std::vector<L1GtLogicParser::OperandToken> &operandTokenVector() { return m_operandTokenVector; }

  void print(std::ostream &myCout) const;

private:
  /// algorithm result
  bool m_algoResult;

  // input
  std::string const &m_logicalExpression;
  RpnVector const &m_rpnVector;

  std::vector<OperandToken> m_operandTokenVector;

  std::vector<CombinationsInCond> m_algoCombinationVector;
};

#endif
