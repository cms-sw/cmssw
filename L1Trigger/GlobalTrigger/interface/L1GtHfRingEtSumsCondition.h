#ifndef GlobalTrigger_L1GtHfRingEtSumsCondition_h
#define GlobalTrigger_L1GtHfRingEtSumsCondition_h

/**
 * \class L1GtHfRingEtSumsCondition
 *
 *
 * Description: evaluation of a CondHfRingEtSums condition.
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

// user include files
//   base classes
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// forward declarations
class L1GtCondition;
class L1GtHfRingEtSumsTemplate;

class L1GlobalTriggerPSB;

// class declaration
class L1GtHfRingEtSumsCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtHfRingEtSumsCondition();

    ///     from base template condition (from event setup usually)
    L1GtHfRingEtSumsCondition(const L1GtCondition*, const L1GlobalTriggerPSB*);

    // copy constructor
    L1GtHfRingEtSumsCondition(const L1GtHfRingEtSumsCondition&);

    // destructor
    virtual ~L1GtHfRingEtSumsCondition();

    // assign operator
    L1GtHfRingEtSumsCondition& operator=(const L1GtHfRingEtSumsCondition&);

public:

    /// the core function to check if the condition matches
     const bool evaluateCondition() const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtHfRingEtSumsTemplate* gtHfRingEtSumsTemplate() const {
        return m_gtHfRingEtSumsTemplate;
    }

    void setGtHfRingEtSumsTemplate(const L1GtHfRingEtSumsTemplate*);

    ///   get / set the pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const {
        return m_gtPSB;
    }

    void setGtPSB(const L1GlobalTriggerPSB*);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtHfRingEtSumsCondition& cp);

private:

    /// pointer to a L1GtHfRingEtSumsTemplate
    const L1GtHfRingEtSumsTemplate* m_gtHfRingEtSumsTemplate;

    /// pointer to PSB, to be able to get the trigger objects
    const L1GlobalTriggerPSB* m_gtPSB;

};

#endif
