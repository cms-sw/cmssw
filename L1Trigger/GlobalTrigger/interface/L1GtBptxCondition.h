#ifndef GlobalTrigger_L1GtBptxCondition_h
#define GlobalTrigger_L1GtBptxCondition_h

/**
 * \class L1GtBptxCondition
 *
 *
 * Description: evaluation of a CondBptx condition.
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
class L1GtBptxTemplate;

// class declaration
class L1GtBptxCondition : public L1GtConditionEvaluation
{

public:

    /// constructors
    ///     default
    L1GtBptxCondition();

    ///     from base template condition (from event setup usually)
    L1GtBptxCondition(const L1GtCondition*, const bool result);

    // copy constructor
    L1GtBptxCondition(const L1GtBptxCondition&);

    // destructor
    virtual ~L1GtBptxCondition();

    // assign operator
    L1GtBptxCondition& operator=(const L1GtBptxCondition&);

public:

    /// the core function to check if the condition matches
    const bool evaluateCondition() const;

    /// print condition
    void print(std::ostream& myCout) const;

public:

    ///   get / set the pointer to a L1GtCondition
    inline const L1GtBptxTemplate* gtBptxTemplate() const {
        return m_gtBptxTemplate;
    }

    void setGtBptxTemplate(const L1GtBptxTemplate*);

    ///   get / set the result
    inline const bool conditionResult() const {
        return m_conditionResult;
    }

    inline void setConditionResult(const bool result) {
        m_conditionResult = result;
    }

private:

    /// copy function for copy constructor and operator=
    void copy(const L1GtBptxCondition& cp);

private:

    /// pointer to a L1GtBptxTemplate
    const L1GtBptxTemplate* m_gtBptxTemplate;

    /// condition result
    bool m_conditionResult;

};

#endif
