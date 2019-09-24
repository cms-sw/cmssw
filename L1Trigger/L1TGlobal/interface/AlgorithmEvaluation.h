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

//   for L1GtLogicParser
#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"

// system include files
#include <iostream>

#include <string>
#include <vector>
#include <map>
#include <stack>
#include <queue>

// if hash map is used

#include <ext/hash_map>
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"

//   how to hash std::string, using a "template specialization"
// DMP Comment out for not to prevent conflicts
namespace __gnu_cxx {

  /** 
      Explicit template specialization of hash of a string class, 
      which just uses the internal char* representation as a wrapper. 
      */
  template <>
  struct hash<std::string> {
    size_t operator()(const std::string& x) const { return hash<const char*>()(x.c_str()); }
  };

}  // namespace __gnu_cxx
// end hash map

// user include files

//   base class
#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"

//

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
    typedef __gnu_cxx ::hash_map<std::string, ConditionEvaluation*> ConditionEvaluationMap;
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
