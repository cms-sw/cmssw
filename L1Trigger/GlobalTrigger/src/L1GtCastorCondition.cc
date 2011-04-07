/**
 * \class L1GtCastorCondition
 *
 *
 * Description: evaluation of a CondCastor condition.
 *
 * Implementation:
 *    Simply put the result read from CASTOR L1 record in the L1GtConditionEvaluation
 *    base class, to be similar with other conditions.
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtCastorCondition.h"

// system include files
#include <iostream>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

// constructors
//     default
L1GtCastorCondition::L1GtCastorCondition() :
    L1GtConditionEvaluation() {

    m_conditionResult = false;

}

//     from base template condition (from event setup usually)
L1GtCastorCondition::L1GtCastorCondition(const L1GtCondition* castorTemplate,
        const bool result) :
            L1GtConditionEvaluation(),
            m_gtCastorTemplate(static_cast<const L1GtCastorTemplate*>(castorTemplate)),
            m_conditionResult(result) {

    // maximum number of objects received for the evaluation of the condition
    // no object
    m_condMaxNumberObjects = 0;

}

// copy constructor
void L1GtCastorCondition::copy(const L1GtCastorCondition &cp) {

    m_gtCastorTemplate = cp.gtCastorTemplate();
    m_conditionResult = cp.conditionResult();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtCastorCondition::L1GtCastorCondition(const L1GtCastorCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtCastorCondition::~L1GtCastorCondition() {

    // empty

}

// equal operator
L1GtCastorCondition& L1GtCastorCondition::operator= (const L1GtCastorCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtCastorCondition::setGtCastorTemplate(
        const L1GtCastorTemplate* castorTemplate) {

    m_gtCastorTemplate = castorTemplate;

}

const bool L1GtCastorCondition::evaluateCondition() const {

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    //
    return m_conditionResult;

}

void L1GtCastorCondition::print(std::ostream& myCout) const {

    m_gtCastorTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

