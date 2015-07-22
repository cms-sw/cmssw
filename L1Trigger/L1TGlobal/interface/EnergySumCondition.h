#ifndef GlobalTrigger_EnergySumCondition_h
#define GlobalTrigger_EnergySumCondition_h

/**
 * \class EnergySumCondition
 * 
 * 
 * Description: evaluation of a CondEnergySum condition.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 *
 */

// system include files
#include <iosfwd>
#include <string>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

// forward declarations
class GtCondition;
class EnergySumTemplate;

namespace l1t {

class L1Candidate;

class GtBoard;

// class declaration
class EnergySumCondition : public ConditionEvaluation
{

public:

    /// constructors
    ///     default
    EnergySumCondition();

    ///     from base template condition (from event setup usually)
    EnergySumCondition(const GtCondition*, const GtBoard*);

    // copy constructor
    EnergySumCondition(const EnergySumCondition&);

    // destructor
    virtual ~EnergySumCondition();

    // assign operator
    EnergySumCondition& operator=(const EnergySumCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const;

    /// print condition
     void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const EnergySumTemplate* gtEnergySumTemplate() const {
        return m_gtEnergySumTemplate;
    }

    void setGtEnergySumTemplate(const EnergySumTemplate*);

    ///   get / set the pointer to uGt GtBoard
    inline const GtBoard* getuGtB() const {
        return m_uGtB;
    }

    void setuGtB(const GtBoard*);

private:

    /// copy function for copy constructor and operator=
    void copy(const EnergySumCondition& cp);

private:

    /// pointer to a EnergySumTemplate
    const EnergySumTemplate* m_gtEnergySumTemplate;

    /// pointer to uGt GtBoard, to be able to get the trigger objects
    const GtBoard* m_uGtB;

};

}
#endif
