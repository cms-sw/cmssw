#ifndef L1Trigger_L1TGlobal_AlgorithmEvaluation_h
#define L1Trigger_L1TGlobal_AlgorithmEvaluation_h

// work-around for missing dependency - force checkout...

/**
 * \class AlgorithmEvaluation
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

#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

// forward declarations
class GlobalAlgorithm;

namespace l1t {

  class ConditionEvaluation;

  // class interface
  class AlgorithmEvaluation {
  public:
    typedef GlobalLogicParser::TokenRPN TokenRPN;
    typedef std::vector<TokenRPN> RpnVector;
    typedef GlobalLogicParser::OperandToken OperandToken;

    /// constructor
    //  AlgorithmEvaluation();

    /// constructor from an algorithm from event setup
    explicit AlgorithmEvaluation(const GlobalAlgorithm&);

    /// copy constructor
    // AlgorithmEvaluation(AlgorithmEvaluation&);

    /// destructor
    // virtual ~AlgorithmEvaluation();

    //typedef std::map<std::string, ConditionEvaluation*> ConditionEvaluationMap;
    typedef std::unordered_map<std::string, ConditionEvaluation*> ConditionEvaluationMap;
    typedef ConditionEvaluationMap::const_iterator CItEvalMap;
    typedef ConditionEvaluationMap::iterator ItEvalMap;

  public:
    /// get / set the result of the algorithm
    inline bool gtAlgoResult() const { return m_algoResult; }

    inline void setGtAlgoResult(const bool algoResult) { m_algoResult = algoResult; }

    /// evaluate an algorithm
    void evaluateAlgorithm(const int chipNumber, const std::vector<ConditionEvaluationMap>&);

    /// get all the object combinations evaluated to true in the conditions
    /// from the algorithm
    inline std::vector<CombinationsInCond>& gtAlgoCombinationVector() { return m_algoCombinationVector; }

    inline std::vector<GlobalLogicParser::OperandToken>& operandTokenVector() { return m_operandTokenVector; }

    void print(std::ostream& myCout) const;

  private:
    /// algorithm result
    bool m_algoResult;

    // input
    std::string const& m_logicalExpression;
    RpnVector const& m_rpnVector;

    std::vector<OperandToken> m_operandTokenVector;

    std::vector<CombinationsInCond> m_algoCombinationVector;
  };

}  // namespace l1t
#endif
