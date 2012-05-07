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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtHfBitCountsCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <vector>

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"

#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// constructors
//     default
L1GtHfBitCountsCondition::L1GtHfBitCountsCondition() :
    L1GtConditionEvaluation() {

    //empty

}

//     from base template condition (from event setup usually)
L1GtHfBitCountsCondition::L1GtHfBitCountsCondition(
        const L1GtCondition* bcTemplate, const L1GlobalTriggerPSB* ptrPSB) :
    L1GtConditionEvaluation(), m_gtHfBitCountsTemplate(
            static_cast<const L1GtHfBitCountsTemplate*> (bcTemplate)), m_gtPSB(
            ptrPSB)
{

    // maximum number of objects received for the evaluation of the condition
    // no objects, in fact, just a count
    m_condMaxNumberObjects = 1;

}

// copy constructor
void L1GtHfBitCountsCondition::copy(const L1GtHfBitCountsCondition &cp) {

    m_gtHfBitCountsTemplate = cp.gtHfBitCountsTemplate();
    m_gtPSB = cp.gtPSB();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtHfBitCountsCondition::L1GtHfBitCountsCondition(const L1GtHfBitCountsCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtHfBitCountsCondition::~L1GtHfBitCountsCondition() {

    // empty

}

// equal operator
L1GtHfBitCountsCondition& L1GtHfBitCountsCondition::operator= (const L1GtHfBitCountsCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtHfBitCountsCondition::setGtHfBitCountsTemplate(const L1GtHfBitCountsTemplate* bcTemplate) {

    m_gtHfBitCountsTemplate = bcTemplate;

}

///   set the pointer to PSB
void L1GtHfBitCountsCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtHfBitCountsCondition::evaluateCondition() const {

    // number of trigger objects in the condition
    // no objects, in fact, just a count
    int iCondition = 0;

    // condition result condResult will be set to true if the HF bit counts
    // passes the requirement
    bool condResult = false;

    // store the index of the HfBitCounts object
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    // get the HF bit counts (event / condition)
    const L1GctHFBitCounts* bitCounts = m_gtPSB->getCandL1HfBitCounts();

    // protection against missing HF bit counts collection
    if (bitCounts == 0) {
        return false;
    }

    const L1GtHfBitCountsTemplate::ObjectParameter objPar =
        ( *(m_gtHfBitCountsTemplate->objectParameter()) )[iCondition];

    // FIXME ask GCT to provide a method to retrieve it
    const unsigned int numberL1HfBitCounts = 4;

    const unsigned int cIndex = objPar.countIndex;
    if (cIndex >= numberL1HfBitCounts) {

        edm::LogError("L1GtHfBitCountsCondition") << "\nL1GtHfBitCountsCondition error: countIndex "
            << cIndex << "greater than GCT maximum index = " << numberL1HfBitCounts
            << "\n  ==> condResult = false " << std::endl;
        return false;

    }

    const unsigned int countValue = bitCounts->bitCount(cIndex);

    // check countThreshold
    if ( !checkThreshold(objPar.countThreshold, countValue, m_gtHfBitCountsTemplate->condGEq()) ) {

        return false;
    }

    // index is always zero - the object is in fact a count
    int indexObj = 0;

    objectsInComb.push_back(indexObj);
    (*m_combinationsInCond).push_back(objectsInComb);

    // if we get here all checks were successful for this combination
    // set the general result for evaluateCondition to "true"

    condResult = true;
    return condResult;

}

void L1GtHfBitCountsCondition::print(std::ostream& myCout) const {

    m_gtHfBitCountsTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

