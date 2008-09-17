#ifndef GlobalTrigger_L1GtHfBitCountsCondition_h
#define GlobalTrigger_L1GtHfBitCountsCondition_h

/**
 * \class L1GtHfBitCountsCondition
 *
 *
 * Description: evaluation of a CondHfBitCounts condition.
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
class L1GtHfBitCountsTemplate;

class L1GlobalTriggerPSB;

// class declaration
class L1GtHfBitCountsCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtHfBitCountsCondition();

    ///     from base template condition (from event setup usually)
    L1GtHfBitCountsCondition(const L1GtCondition*, const L1GlobalTriggerPSB*);

    // copy constructor
    L1GtHfBitCountsCondition(const L1GtHfBitCountsCondition&);

    // destructor
    virtual ~L1GtHfBitCountsCondition();

    // assign operator
    L1GtHfBitCountsCondition& operator=(const L1GtHfBitCountsCondition&);

public:

    /// the core function to check if the condition matches
     const bool evaluateCondition() const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtHfBitCountsTemplate* gtHfBitCountsTemplate() const {
        return m_gtHfBitCountsTemplate;
    }

    void setGtHfBitCountsTemplate(const L1GtHfBitCountsTemplate*);

    ///   get / set the pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const {
        return m_gtPSB;
    }

    void setGtPSB(const L1GlobalTriggerPSB*);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtHfBitCountsCondition& cp);

private:

    /// pointer to a L1GtHfBitCountsTemplate
    const L1GtHfBitCountsTemplate* m_gtHfBitCountsTemplate;

    /// pointer to PSB, to be able to get the trigger objects
    const L1GlobalTriggerPSB* m_gtPSB;

};

#endif
