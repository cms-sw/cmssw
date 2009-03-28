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

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtEnergySumCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// constructors
//     default
L1GtEnergySumCondition::L1GtEnergySumCondition() :
    L1GtConditionEvaluation() {

    //empty

}

//     from base template condition (from event setup usually)
L1GtEnergySumCondition::L1GtEnergySumCondition(const L1GtCondition* eSumTemplate,
    const L1GlobalTriggerPSB* ptrPSB) :
    L1GtConditionEvaluation(),
    m_gtEnergySumTemplate(static_cast<const L1GtEnergySumTemplate*>(eSumTemplate)),
    m_gtPSB(ptrPSB)

{

    // maximum number of objects received for the evaluation of the condition
    // energy sums are global quantities - one object per event

    m_condMaxNumberObjects = 1;

}

// copy constructor
void L1GtEnergySumCondition::copy(const L1GtEnergySumCondition &cp) {

    m_gtEnergySumTemplate = cp.gtEnergySumTemplate();
    m_gtPSB = cp.gtPSB();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtEnergySumCondition::L1GtEnergySumCondition(const L1GtEnergySumCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtEnergySumCondition::~L1GtEnergySumCondition() {

    // empty

}

// equal operator
L1GtEnergySumCondition& L1GtEnergySumCondition::operator= (const L1GtEnergySumCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtEnergySumCondition::setGtEnergySumTemplate(
    const L1GtEnergySumTemplate* eSumTempl) {

    m_gtEnergySumTemplate = eSumTempl;

}

///   set the pointer to PSB
void L1GtEnergySumCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtEnergySumCondition::evaluateCondition() const {

    // number of trigger objects in the condition
    // in fact, there is only one object
    int iCondition = 0;

    // condition result condResult set to true if the energy sum
    // passes all requirements
    bool condResult = false;

    // store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    // clear the indices in the combination
    objectsInComb.clear();

    // get energy, phi (ETM and HTM) and overflow for the trigger object

    unsigned int candEt = 0;
    unsigned int candPhi = 0;
    bool candOverflow = false;

    switch ((m_gtEnergySumTemplate->objectType())[0]) {
        case ETT: {
            const L1GctEtTotal* cand1 = m_gtPSB->getCandL1ETT();

            if (cand1 == 0) {
                return false;
            }

            candEt = cand1->et();
            candOverflow = cand1->overFlow();

            break;
        }
        case ETM: {
            const L1GctEtMiss* cand2 = m_gtPSB->getCandL1ETM();

            if (cand2 == 0) {
                return false;
            }

            candEt = cand2->et();
            candPhi = cand2->phi();
            candOverflow = cand2->overFlow();

            break;
        }
        case HTT: {
            const L1GctEtHad* cand3 = m_gtPSB->getCandL1HTT();

            if (cand3 == 0) {
                return false;
            }

            candEt = cand3->et();
            candOverflow = cand3->overFlow();

            break;
        }
        case HTM: {
            const L1GctHtMiss* cand4 = m_gtPSB->getCandL1HTM();

            if (cand4 == 0) {
                return false;
            }

            candEt = cand4->et();
            candPhi = cand4->phi();
            candOverflow = cand4->overFlow();

            break;
        }
        default: {
            // should not arrive here
            return false;

            break;
        }
    }

    const L1GtEnergySumTemplate::ObjectParameter objPar =
        ( *(m_gtEnergySumTemplate->objectParameter()) )[iCondition];

    // check energy threshold and overflow
    // overflow evaluation:
    //     for condGEq >=
    //         candidate overflow true -> condition true
    //         candidate overflow false -> evaluate threshold
    //     for condGEq =
    //         candidate overflow true -> condition false
    //         candidate overflow false -> evaluate threshold
    //

    bool condGEqVal = m_gtEnergySumTemplate->condGEq();

    if (condGEqVal) {
        if (!candOverflow) {
            if (!checkThreshold(objPar.etThreshold, candEt, condGEqVal)) {
                return false;
            }
        }
    } else {
        if (candOverflow) {
            return false;
        } else {
            if (!checkThreshold(objPar.etThreshold, candEt, condGEqVal)) {
                return false;
            }
        }

    }

    // for ETM and HTM check phi also
    // for overflow, the phi requirements are ignored

    if (!candOverflow) {
        if ( ( m_gtEnergySumTemplate->objectType() )[0] == ETM) {

            // phi bitmask is saved in two uint64_t (see parser)
            if (candPhi < 64) {
                if (!checkBit(objPar.phiRange0Word, candPhi)) {

                    return false;
                }
            } else {
                if (!checkBit(objPar.phiRange1Word, candPhi - 64)) {

                    return false;
                }
            }

        } else if ( ( m_gtEnergySumTemplate->objectType() )[0] == HTM) {

            // phi bitmask is in the first word for HTM
            if (candPhi < 64) {
                if (!checkBit(objPar.phiRange0Word, candPhi)) {

                    return false;
                }
            } else {
                if (!checkBit(objPar.phiRange1Word, candPhi - 64)) {

                    return false;
                }
            }
        }
    }


    // index is always zero, as they are global quantities (there is only one object)
    int indexObj = 0;

    objectsInComb.push_back(indexObj);
    (*m_combinationsInCond).push_back(objectsInComb);

    // if we get here all checks were successfull for this combination
    // set the general result for evaluateCondition to "true"

    condResult = true;
    return condResult;

}

void L1GtEnergySumCondition::print(std::ostream& myCout) const {

    m_gtEnergySumTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

