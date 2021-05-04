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

//   for L1GtLogicParser
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

// system include files
#include <iostream>

#include <map>
#include <queue>
#include <stack>
#include <string>
#include <vector>

// if hash map is used

#include <ext/hash_map>

//   how to hash std::string, using a "template specialization"
namespace __gnu_cxx {

  /**
 Explicit template specialization of hash of a string class,
 which just uses the internal char* representation as a wrapper.
 */
  template <>
  struct hash<std::string> {
    size_t operator()(const std::string &x) const { return hash<const char *>()(x.c_str()); }
  };

}  // namespace __gnu_cxx
// end hash map

// user include files

//   base class
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

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
  typedef __gnu_cxx ::hash_map<std::string, L1GtConditionEvaluation *> ConditionEvaluationMap;
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
