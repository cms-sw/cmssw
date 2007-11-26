#ifndef GlobalTrigger_L1GtCaloCondition_h
#define GlobalTrigger_L1GtCaloCondition_h

/**
 * \class L1GtCaloCondition
 * 
 * 
 * Description: evaluation of a CondCalo condition.
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
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

// forward declarations

// class declaration
class L1GtCaloCondition :
            public L1GtCaloTemplate, public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtCaloCondition();

    ///     from name of the condition
    L1GtCaloCondition(const std::string&);

    ///     from base template condition (from event setup usually)
    L1GtCaloCondition(const L1GtCaloTemplate&);

    // copy constructor
    L1GtCaloCondition(const L1GtCaloCondition&);

    // destructor
    virtual ~L1GtCaloCondition();

    // assign operator
    L1GtCaloCondition& operator= (const L1GtCaloCondition&);

public:

    /// the core function to check if the condition matches
    virtual const bool evaluateCondition() const;

    /// print condition
    virtual void print(std::ostream& myCout) const;

public:

    ///   get / set the number of bits for eta of calorimeter objects
    inline int gtIfCaloEtaNumberBits() const
    {
        return m_ifCaloEtaNumberBits;
    }

    void setGtIfCaloEtaNumberBits(const int&);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtCaloCondition& cp);

    /// load calo candidates
    virtual L1GctCand* getCandidate(int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool checkObjectParameter(const int iCondition, const L1GctCand& cand) const;

private:

    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

};

#endif
