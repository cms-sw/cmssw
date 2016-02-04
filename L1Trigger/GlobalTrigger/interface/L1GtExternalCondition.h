#ifndef GlobalTrigger_L1GtExternalCondition_h
#define GlobalTrigger_L1GtExternalCondition_h

/**
 * \class L1GtExternalCondition
 *
 *
 * Description: evaluation of a CondExternal condition.
 *
 * Implementation:
 *    Simply put the result read in the L1GtConditionEvaluation
 *    base class, to be similar with other conditions.
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
class L1GtExternalTemplate;

// class declaration
class L1GtExternalCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtExternalCondition();

    ///     from base template condition (from event setup usually)
    L1GtExternalCondition(const L1GtCondition*, const bool result);

    // copy constructor
    L1GtExternalCondition(const L1GtExternalCondition&);

    // destructor
    virtual ~L1GtExternalCondition();

    // assign operator
    L1GtExternalCondition& operator=(const L1GtExternalCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition() const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtExternalTemplate* gtExternalTemplate() const {
        return m_gtExternalTemplate;
    }

    void setGtExternalTemplate(const L1GtExternalTemplate*);

    ///   get / set the result
    inline const bool conditionResult() const {
        return m_conditionResult;
    }

    inline void setConditionResult(const bool result) {
        m_conditionResult = result;
    }

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtExternalCondition& cp);

private:

    /// pointer to a L1GtExternalTemplate
    const L1GtExternalTemplate* m_gtExternalTemplate;

    /// condition result
    bool m_conditionResult;

};

#endif
