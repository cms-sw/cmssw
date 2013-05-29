/**
 * \class L1GtHfRingEtSumsCondition
 *
 *
 * Description: evaluation of a CondHfRingEtSums condition.
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
#include "L1Trigger/GlobalTrigger/interface/L1GtHfRingEtSumsCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <vector>

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"

#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// constructors
//     default
L1GtHfRingEtSumsCondition::L1GtHfRingEtSumsCondition() :
    L1GtConditionEvaluation() {

    //empty

}

//     from base template condition (from event setup usually)
L1GtHfRingEtSumsCondition::L1GtHfRingEtSumsCondition(
        const L1GtCondition* etTemplate, const L1GlobalTriggerPSB* ptrPSB) :
    L1GtConditionEvaluation(), m_gtHfRingEtSumsTemplate(
            static_cast<const L1GtHfRingEtSumsTemplate*> (etTemplate)), m_gtPSB(
            ptrPSB)
{

    // maximum number of objects received for the evaluation of the condition
    // no objects, in fact, just a number
    m_condMaxNumberObjects = 1;

}

// copy constructor
void L1GtHfRingEtSumsCondition::copy(const L1GtHfRingEtSumsCondition &cp) {

    m_gtHfRingEtSumsTemplate = cp.gtHfRingEtSumsTemplate();
    m_gtPSB = cp.gtPSB();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtHfRingEtSumsCondition::L1GtHfRingEtSumsCondition(const L1GtHfRingEtSumsCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtHfRingEtSumsCondition::~L1GtHfRingEtSumsCondition() {

    // empty

}

// equal operator
L1GtHfRingEtSumsCondition& L1GtHfRingEtSumsCondition::operator= (const L1GtHfRingEtSumsCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtHfRingEtSumsCondition::setGtHfRingEtSumsTemplate(const L1GtHfRingEtSumsTemplate* etTemplate) {

    m_gtHfRingEtSumsTemplate = etTemplate;

}

///   set the pointer to PSB
void L1GtHfRingEtSumsCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtHfRingEtSumsCondition::evaluateCondition() const {

    // number of trigger objects in the condition
    // no objects, in fact, just a number
    int iCondition = 0;

    // condition result condResult will be set to true if the HF Ring Et sums
    // passes the requirement
    bool condResult = false;

    // store the index of the HfRingEtSums object
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    // get the HF Ring Et sums (event / condition)
    const L1GctHFRingEtSums* etSumCand = m_gtPSB->getCandL1HfRingEtSums();

    // protection against missing HF Ring Et sums collection
    if (etSumCand == 0) {
        return false;
    }

    const L1GtHfRingEtSumsTemplate::ObjectParameter objPar =
        ( *(m_gtHfRingEtSumsTemplate->objectParameter()) )[iCondition];

    // FIXME ask GCT to provide a method to retrieve it
    const unsigned int numberL1HfRingEtSums = 4;

    const unsigned int cIndex = objPar.etSumIndex;
    if (cIndex >= numberL1HfRingEtSums) {

        edm::LogError("L1GtHfRingEtSumsCondition") << "\nL1GtHfRingEtSumsCondition error: etSumIndex "
            << cIndex << "greater than GCT maximum index = " << numberL1HfRingEtSums
            << "\n  ==> condResult = false " << std::endl;
        return false;

    }

    const unsigned int etSumValue = etSumCand->etSum(cIndex);

    // check countThreshold
    if ( !checkThreshold(objPar.etSumThreshold, etSumValue, m_gtHfRingEtSumsTemplate->condGEq()) ) {

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

void L1GtHfRingEtSumsCondition::print(std::ostream& myCout) const {

    m_gtHfRingEtSumsTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

