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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtBptxCondition.h"

// system include files
#include <iostream>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// constructors
//     default
L1GtBptxCondition::L1GtBptxCondition() :
    L1GtConditionEvaluation() {

    m_conditionResult = false;

}

//     from base template condition (from event setup usually)
L1GtBptxCondition::L1GtBptxCondition(const L1GtCondition* bptxTemplate,
        const bool result) :
            L1GtConditionEvaluation(),
            m_gtBptxTemplate(static_cast<const L1GtBptxTemplate*>(bptxTemplate)),
            m_conditionResult(result) {

    // maximum number of objects received for the evaluation of the condition
    // no object
    m_condMaxNumberObjects = 0;

}

// copy constructor
void L1GtBptxCondition::copy(const L1GtBptxCondition &cp) {

    m_gtBptxTemplate = cp.gtBptxTemplate();
    m_conditionResult = cp.conditionResult();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtBptxCondition::L1GtBptxCondition(const L1GtBptxCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtBptxCondition::~L1GtBptxCondition() {

    // empty

}

// equal operator
L1GtBptxCondition& L1GtBptxCondition::operator= (const L1GtBptxCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtBptxCondition::setGtBptxTemplate(
        const L1GtBptxTemplate* bptxTemplate) {

    m_gtBptxTemplate = bptxTemplate;

}

const bool L1GtBptxCondition::evaluateCondition() const {

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    //
    return m_conditionResult;

}

void L1GtBptxCondition::print(std::ostream& myCout) const {

    m_gtBptxTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

