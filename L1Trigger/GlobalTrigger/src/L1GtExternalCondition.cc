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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtExternalCondition.h"

// system include files
#include <iostream>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// constructors
//     default
L1GtExternalCondition::L1GtExternalCondition() :
    L1GtConditionEvaluation() {

    m_conditionResult = false;

}

//     from base template condition (from event setup usually)
L1GtExternalCondition::L1GtExternalCondition(const L1GtCondition* externalTemplate,
        const bool result) :
            L1GtConditionEvaluation(),
            m_gtExternalTemplate(static_cast<const L1GtExternalTemplate*>(externalTemplate)),
            m_conditionResult(result) {

    // maximum number of objects received for the evaluation of the condition
    // no object
    m_condMaxNumberObjects = 0;

}

// copy constructor
void L1GtExternalCondition::copy(const L1GtExternalCondition &cp) {

    m_gtExternalTemplate = cp.gtExternalTemplate();
    m_conditionResult = cp.conditionResult();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtExternalCondition::L1GtExternalCondition(const L1GtExternalCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtExternalCondition::~L1GtExternalCondition() {

    // empty

}

// equal operator
L1GtExternalCondition& L1GtExternalCondition::operator= (const L1GtExternalCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtExternalCondition::setGtExternalTemplate(
        const L1GtExternalTemplate* externalTemplate) {

    m_gtExternalTemplate = externalTemplate;

}

const bool L1GtExternalCondition::evaluateCondition() const {

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    //
    return m_conditionResult;

}

void L1GtExternalCondition::print(std::ostream& myCout) const {

    m_gtExternalTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

