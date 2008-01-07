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
#include <iomanip>

#include <string>
#include <vector>
#include <map>

#include <boost/cstdint.hpp>

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
    L1GtAlgorithmEvaluation(const L1GtAlgorithm*);

    /// copy constructor
    L1GtAlgorithmEvaluation(L1GtAlgorithmEvaluation&);

    /// destructor
    virtual ~L1GtAlgorithmEvaluation();

public:

    /// get / set the result of the algorithm
    inline const bool& gtAlgoResult() const {
        return m_algoResult;
    }

    inline void setGtAlgoResult(const bool algoResult) {
        m_algoResult = algoResult;
    }

    /// evaluate an algorithm
    void evaluateAlgorithm(const int chipNumber, 
        const std::vector<std::map<std::string, L1GtConditionEvaluation*> >&);

    /// get the numeric expression
    inline const std::string& gtAlgoNumericalExpression() const {
        return m_algoNumericalExpression;

    }

    /// get all the object combinations evaluated to true in the conditions 
    /// from the algorithm 
    inline const std::vector<CombinationsInCond>* gtAlgoCombinationVector() const {
        return &m_algoCombinationVector;
    }
    
    void print(std::ostream& myCout) const;


private:

    /// algorithm result
    bool m_algoResult;

    /// algorithm numerical expresssion
    std::string m_algoNumericalExpression;

    std::vector<CombinationsInCond> m_algoCombinationVector;

};

#endif
