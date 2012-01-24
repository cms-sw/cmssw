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
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// forward declarations
class L1GtCondition;
class L1GtCaloTemplate;

class L1GctCand;

class L1GlobalTriggerPSB;

// class declaration
class L1GtCaloCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtCaloCondition();

    ///     from base template condition (from event setup usually)
    L1GtCaloCondition(const L1GtCondition*, const L1GlobalTriggerPSB*,
            const int nrL1NoIsoEG,
            const int nrL1IsoEG,
            const int nrL1CenJet,
            const int nrL1ForJet,
            const int nrL1TauJet,
            const int ifCaloEtaNumberBits);

    // copy constructor
    L1GtCaloCondition(const L1GtCaloCondition&);

    // destructor
    virtual ~L1GtCaloCondition();

    // assign operator
    L1GtCaloCondition& operator=(const L1GtCaloCondition&);

public:

    /// the core function to check if the condition matches
     const bool evaluateCondition() const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtCaloTemplate* gtCaloTemplate() const {
        return m_gtCaloTemplate;
    }

    void setGtCaloTemplate(const L1GtCaloTemplate*);

    ///   get / set the number of bits for eta of calorimeter objects
    inline const int gtIfCaloEtaNumberBits() const {
        return m_ifCaloEtaNumberBits;
    }

    void setGtIfCaloEtaNumberBits(const int&);

    ///   get / set the pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const {
        return m_gtPSB;
    }

    void setGtPSB(const L1GlobalTriggerPSB*);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtCaloCondition& cp);

    /// load calo candidates
    const L1GctCand* getCandidate(const int indexCand) const;

    /// function to check a single object if it matches a condition
    const bool
        checkObjectParameter(const int iCondition, const L1GctCand& cand) const;

private:

    /// pointer to a L1GtCaloTemplate
    const L1GtCaloTemplate* m_gtCaloTemplate;

    /// pointer to PSB, to be able to get the trigger objects
    const L1GlobalTriggerPSB* m_gtPSB;

    /// number of bits for eta of calorimeter objects
    int m_ifCaloEtaNumberBits;

};

#endif
