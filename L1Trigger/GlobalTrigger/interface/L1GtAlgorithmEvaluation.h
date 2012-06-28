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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iostream>

#include <string>
#include <vector>
#include <map>

#include <boost/cstdint.hpp>

// if hash map is used

#include <ext/hash_map>

//   how to hash std::string, using a "template specialization"
namespace __gnu_cxx
{

/**
 Explicit template specialization of hash of a string class,
 which just uses the internal char* representation as a wrapper.
 */
template <> struct hash<std::string>
{
    size_t operator()(const std::string& x) const {
        return hash<const char*>()(x.c_str());
    }
};

}
// end hash map


// user include files

//   base class
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations
class L1GtAlgorithm;
class L1GtConditionEvaluation;

// class interface
class L1GtAlgorithmEvaluation : public L1GtLogicParser
{

public:

    /// constructor
    L1GtAlgorithmEvaluation();

    /// constructor from an algorithm from event setup
    L1GtAlgorithmEvaluation(const L1GtAlgorithm&);

    /// copy constructor
    L1GtAlgorithmEvaluation(L1GtAlgorithmEvaluation&);

    /// destructor
    virtual ~L1GtAlgorithmEvaluation();
    
    //typedef std::map<std::string, L1GtConditionEvaluation*> ConditionEvaluationMap;
    typedef __gnu_cxx::hash_map<std::string, L1GtConditionEvaluation*> ConditionEvaluationMap;
    typedef ConditionEvaluationMap::const_iterator CItEvalMap ;
    typedef ConditionEvaluationMap::iterator ItEvalMap  ;

public:

    /// get / set the result of the algorithm
    inline const bool& gtAlgoResult() const {
        return m_algoResult;
    }

    inline void setGtAlgoResult(const bool algoResult) {
        m_algoResult = algoResult;
    }

    /// evaluate an algorithm
    void evaluateAlgorithm(const int chipNumber, const std::vector<ConditionEvaluationMap>&);

    /// get all the object combinations evaluated to true in the conditions 
    /// from the algorithm 
    inline const std::vector<CombinationsInCond>* gtAlgoCombinationVector() const {
        return &m_algoCombinationVector;
    }
    
    void print(std::ostream& myCout) const;


private:

    /// algorithm result
    bool m_algoResult;

    std::vector<CombinationsInCond> m_algoCombinationVector;

};

#endif
