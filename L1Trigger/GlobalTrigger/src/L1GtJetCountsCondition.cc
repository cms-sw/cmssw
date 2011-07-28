/**
 * \class L1GtJetCountsCondition
 *
 *
 * Description: evaluation of a CondJetCounts condition.
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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtJetCountsCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <vector>

// user include files
//   base classes
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// constructors
//     default
L1GtJetCountsCondition::L1GtJetCountsCondition() :
    L1GtConditionEvaluation() {

    //empty

}

//     from base template condition (from event setup usually)
L1GtJetCountsCondition::L1GtJetCountsCondition(const L1GtCondition* jcTemplate,
    const L1GlobalTriggerPSB* ptrPSB, const int nrL1JetCounts) :
    L1GtConditionEvaluation(),
    m_gtJetCountsTemplate(static_cast<const L1GtJetCountsTemplate*>(jcTemplate)),
    m_gtPSB(ptrPSB),
    m_numberL1JetCounts(nrL1JetCounts)
{

    // maximum number of objects received for the evaluation of the condition
    // no objects, in fact, just a number
    m_condMaxNumberObjects = 1;

}

// copy constructor
void L1GtJetCountsCondition::copy(const L1GtJetCountsCondition &cp) {

    m_gtJetCountsTemplate = cp.gtJetCountsTemplate();
    m_gtPSB = cp.gtPSB();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtJetCountsCondition::L1GtJetCountsCondition(const L1GtJetCountsCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtJetCountsCondition::~L1GtJetCountsCondition() {

    // empty

}

// equal operator
L1GtJetCountsCondition& L1GtJetCountsCondition::operator= (const L1GtJetCountsCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtJetCountsCondition::setGtJetCountsTemplate(const L1GtJetCountsTemplate* jcTemplate) {

    m_gtJetCountsTemplate = jcTemplate;

}

///   set the pointer to PSB
void L1GtJetCountsCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtJetCountsCondition::evaluateCondition() const {

    // number of trigger objects in the condition
    // in fact, there is only one object
    int iCondition = 0;

    // condition result condResult will be set to true if the jet counts
    // passes the requirement
    bool condResult = false;

    // store the index of the JetCount object
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    // get the jet counts (event / condition)
    const L1GctJetCounts* jetCounts = m_gtPSB->getCandL1JetCounts();

    // protection against missing jet counts collection
    if (jetCounts == 0) {
        return false;
    }

    const L1GtJetCountsTemplate::ObjectParameter objPar =
        ( *(m_gtJetCountsTemplate->objectParameter()) )[iCondition];

    unsigned int cIndex = objPar.countIndex;

    if (cIndex >= m_numberL1JetCounts) {

        edm::LogError("L1GtJetCountsCondition") << "\nL1GtJetCountsCondition error: countIndex "
            << cIndex << "greater than maximum allowed count = " << m_numberL1JetCounts
            << "\n  ==> condResult = false " << std::endl;
        return false;

    }

    unsigned int countValue = jetCounts->count(cIndex);

    // check countThreshold
    if ( !checkThreshold(objPar.countThreshold, countValue, m_gtJetCountsTemplate->condGEq()) ) {

        return false;
    }

    // index is always zero, as they are global quantities (there is only one object)
    int indexObj = 0;

    objectsInComb.push_back(indexObj);
    (*m_combinationsInCond).push_back(objectsInComb);

    // if we get here all checks were successful for this combination
    // set the general result for evaluateCondition to "true"

    condResult = true;
    return condResult;

}

void L1GtJetCountsCondition::print(std::ostream& myCout) const {

    m_gtJetCountsTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

