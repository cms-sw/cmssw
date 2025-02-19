#ifndef GlobalTrigger_L1GtEnergySumCondition_h
#define GlobalTrigger_L1GtEnergySumCondition_h

/**
 * \class L1GtEnergySumCondition
 * 
 * 
 * Description: evaluation of a CondEnergySum condition.
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
class L1GtEnergySumTemplate;

class L1GlobalTriggerPSB;

// class declaration
class L1GtEnergySumCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtEnergySumCondition();

    ///     from base template condition (from event setup usually)
    L1GtEnergySumCondition(const L1GtCondition*, const L1GlobalTriggerPSB*);

    // copy constructor
    L1GtEnergySumCondition(const L1GtEnergySumCondition&);

    // destructor
    virtual ~L1GtEnergySumCondition();

    // assign operator
    L1GtEnergySumCondition& operator=(const L1GtEnergySumCondition&);

public:

    /// the core function to check if the condition matches
     const bool evaluateCondition() const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtEnergySumTemplate* gtEnergySumTemplate() const {
        return m_gtEnergySumTemplate;
    }

    void setGtEnergySumTemplate(const L1GtEnergySumTemplate*);

    ///   get / set the pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const {
        return m_gtPSB;
    }

    void setGtPSB(const L1GlobalTriggerPSB*);

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtEnergySumCondition& cp);

private:

    /// pointer to a L1GtEnergySumTemplate
    const L1GtEnergySumTemplate* m_gtEnergySumTemplate;

    /// pointer to PSB, to be able to get the trigger objects
    const L1GlobalTriggerPSB* m_gtPSB;

};

#endif
