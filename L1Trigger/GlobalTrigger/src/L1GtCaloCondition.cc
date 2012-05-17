/**
 * \class L1GtCaloCondition
 *
 *
 * Description: evaluation of a CondCalo condition.
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
#include "L1Trigger/GlobalTrigger/interface/L1GtCaloCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
L1GtCaloCondition::L1GtCaloCondition() :
    L1GtConditionEvaluation() {

    m_ifCaloEtaNumberBits = -1;

}

//     from base template condition (from event setup usually)
L1GtCaloCondition::L1GtCaloCondition(const L1GtCondition* caloTemplate, const L1GlobalTriggerPSB* ptrPSB,
        const int nrL1NoIsoEG,
        const int nrL1IsoEG,
        const int nrL1CenJet,
        const int nrL1ForJet,
        const int nrL1TauJet,
        const int ifCaloEtaNumberBits) :
    L1GtConditionEvaluation(),
    m_gtCaloTemplate(static_cast<const L1GtCaloTemplate*>(caloTemplate)),
    m_gtPSB(ptrPSB),
    m_ifCaloEtaNumberBits(ifCaloEtaNumberBits)
{

    // maximum number of objects received for the evaluation of the condition
    // retrieved before from event setup
    // for a CondCalo, all objects ar of same type, hence it is enough to get the
    // type for the first object

    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            m_condMaxNumberObjects = nrL1NoIsoEG;
            break;
        case IsoEG:
            m_condMaxNumberObjects = nrL1IsoEG;
            break;
        case CenJet:
            m_condMaxNumberObjects = nrL1CenJet;
            break;
        case ForJet:
            m_condMaxNumberObjects = nrL1ForJet;
            break;
        case TauJet:
            m_condMaxNumberObjects = nrL1TauJet;
            break;
        default:
            m_condMaxNumberObjects = 0;
            break;
    }

}

// copy constructor
void L1GtCaloCondition::copy(const L1GtCaloCondition &cp) {

    m_gtCaloTemplate = cp.gtCaloTemplate();
    m_gtPSB = cp.gtPSB();

    m_ifCaloEtaNumberBits = cp.gtIfCaloEtaNumberBits();

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

L1GtCaloCondition::L1GtCaloCondition(const L1GtCaloCondition& cp) :
    L1GtConditionEvaluation() {

    copy(cp);

}

// destructor
L1GtCaloCondition::~L1GtCaloCondition() {

    // empty

}

// equal operator
L1GtCaloCondition& L1GtCaloCondition::operator= (const L1GtCaloCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void L1GtCaloCondition::setGtCaloTemplate(const L1GtCaloTemplate* caloTempl) {

    m_gtCaloTemplate = caloTempl;

}

//   set the number of bits for eta of calorimeter objects
void L1GtCaloCondition::setGtIfCaloEtaNumberBits(const int& ifCaloEtaNumberBitsValue) {

    m_ifCaloEtaNumberBits = ifCaloEtaNumberBitsValue;

}

///   set the pointer to PSB
void L1GtCaloCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtCaloCondition::evaluateCondition() const {

    // number of trigger objects in the condition
    int nObjInCond = m_gtCaloTemplate->nrObjects();
    //LogTrace("L1GtCaloCondition") << "  nObjInCond: " << nObjInCond
    //    << std::endl;

    // the candidates

    // objectType() gives the type for nrObjects() only,
    // but in a CondCalo all objects have the same type
    // take type from the type of the first object

    const std::vector<const L1GctCand*>* candVec;

    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            candVec = m_gtPSB->getCandL1NoIsoEG();
            break;
        case IsoEG:
            candVec = m_gtPSB->getCandL1IsoEG();
            break;
        case CenJet:
            candVec = m_gtPSB->getCandL1CenJet();
            break;
        case ForJet:
            candVec = m_gtPSB->getCandL1ForJet();
            break;
        case TauJet:
            candVec = m_gtPSB->getCandL1TauJet();
            break;
        default:
            return false;
            break;
    }

    int numberObjects = candVec->size();
    //LogTrace("L1GtCaloCondition") << "  numberObjects: " << numberObjects
    //    << std::endl;
    if (numberObjects < nObjInCond) {
        return false;
    }

    std::vector<int> index(numberObjects);

    for (int i = 0; i < numberObjects; ++i) {
        index[i] = i;
    }

    int jumpIndex = 1;
    int jump = factorial(numberObjects - nObjInCond);

    int totalLoops = 0;
    int passLoops = 0;

    // condition result condResult set to true if at least one permutation
    //     passes all requirements
    // all possible permutations are checked
    bool condResult = false;

    // store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;
    objectsInComb.reserve(nObjInCond);

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    do {

        if (--jumpIndex)
            continue;

        jumpIndex = jump;
        totalLoops++;

        // clear the indices in the combination
        objectsInComb.clear();

        bool tmpResult = true;

        // check if there is a permutation that matches object-parameter requirements
        for (int i = 0; i < nObjInCond; i++) {

            tmpResult &= checkObjectParameter(i, *(*candVec)[index[i]]);
            objectsInComb.push_back(index[i]);

        }

        // if permutation does not match particle conditions
        // skip spatial correlations
        if (!tmpResult) {

            continue;

        }

        if (m_gtCaloTemplate->wsc()) {

            // wsc requirements have always nObjInCond = 2
            // one can use directly index[0] and index[1] to compute
            // eta and phi differences
            const int ObjInWscComb = 2;
            if (nObjInCond != ObjInWscComb) {

                if (m_verbosity) {
                    edm::LogError("L1GtCaloCondition")
                        << "\n  Error: "
                        << "number of particles in condition with spatial correlation = "
                        << nObjInCond << "\n  it must be = " << ObjInWscComb
                        << std::endl;
                }

                continue;
            }

            L1GtCaloTemplate::CorrelationParameter corrPar =
                *(m_gtCaloTemplate->correlationParameter());

            unsigned int candDeltaEta;
            unsigned int candDeltaPhi;

            // check candDeltaEta

            // get eta index and the sign bit of the eta index (MSB is the sign)
            //   signedEta[i] is the signed eta index of candVec[index[i]]
            int signedEta[ObjInWscComb];
            int signBit[ObjInWscComb] = { 0, 0 };

            int scaleEta = 1 << (m_ifCaloEtaNumberBits - 1);

            for (int i = 0; i < ObjInWscComb; ++i) {
                signBit[i] = ((*candVec)[index[i]]->etaIndex() & scaleEta)
                    >>(m_ifCaloEtaNumberBits - 1);
                signedEta[i] = ((*candVec)[index[i]]->etaIndex() )%scaleEta;

                if (signBit[i] == 1) {
                    signedEta[i] = (-1)*signedEta[i];
                }

            }

            // compute candDeltaEta - add 1 if signs are different (due to +0/-0 indices)
            candDeltaEta = static_cast<int> (std::abs(signedEta[1] - signedEta[0]))
                + static_cast<int> (signBit[1]^signBit[0]);

            if ( !checkBit(corrPar.deltaEtaRange, candDeltaEta) ) {
                continue;
            }

            // check candDeltaPhi

            // calculate absolute value of candDeltaPhi
            if ((*candVec)[index[0]]->phiIndex()> (*candVec)[index[1]]->phiIndex()) {
                candDeltaPhi = (*candVec)[index[0]]->phiIndex() - (*candVec)[index[1]]->phiIndex();
            }
            else {
                candDeltaPhi = (*candVec)[index[1]]->phiIndex() - (*candVec)[index[0]]->phiIndex();
            }

            // check if candDeltaPhi > 180 (via delta_phi_maxbits)
            // delta_phi contains bits for 0..180 (0 and 180 included)
            while (candDeltaPhi> corrPar.deltaPhiMaxbits) {

                // candDeltaPhi > 180 ==> take 360 - candDeltaPhi
                candDeltaPhi = (corrPar.deltaPhiMaxbits - 1)*2 - candDeltaPhi;
                if (m_verbosity) {
                    LogTrace("L1GtCaloCondition")
                        << "  candDeltaPhi rescaled to: " << candDeltaPhi
                        << std::endl;
                }
            }

            if (!checkBit(corrPar.deltaPhiRange, candDeltaPhi)) {
                continue;
            }

        } // end wsc check

        // if we get here all checks were successfull for this combination
        // set the general result for evaluateCondition to "true"

        condResult = true;
        passLoops++;
        (*m_combinationsInCond).push_back(objectsInComb);

        //    } while ( std::next_permutation(index, index + nObj) );
    } while (std::next_permutation(index.begin(), index.end()) );

    //LogTrace("L1GtCaloCondition")
    //    << "\n  L1GtCaloCondition: total number of permutations found:          " << totalLoops
    //    << "\n  L1GtCaloCondition: number of permutations passing requirements: " << passLoops
    //    << "\n" << std::endl;

    return condResult;

}

// load calo candidates
const L1GctCand* L1GtCaloCondition::getCandidate(const int indexCand) const {

    // objectType() gives the type for nrObjects() only,
    // but in a CondCalo all objects have the same type
    // take type from the type of the first object
    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            return (*(m_gtPSB->getCandL1NoIsoEG()))[indexCand];
            break;
        case IsoEG:
            return (*(m_gtPSB->getCandL1IsoEG()))[indexCand];
            break;
        case CenJet:
            return (*(m_gtPSB->getCandL1CenJet()))[indexCand];
            break;
        case ForJet:
            return (*(m_gtPSB->getCandL1ForJet()))[indexCand];
            break;
        case TauJet:
            return (*(m_gtPSB->getCandL1TauJet()))[indexCand];
            break;
        default:
            return 0;
            break;
    }

    return 0;
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition.
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool L1GtCaloCondition::checkObjectParameter(const int iCondition, const L1GctCand& cand) const {

    // number of objects in condition
    int nObjInCond = m_gtCaloTemplate->nrObjects();

    if (iCondition >= nObjInCond || iCondition < 0) {
        return false;
    }

    // empty candidates can not be compared
    if (cand.empty()) {
        return false;
    }

    const L1GtCaloTemplate::ObjectParameter objPar = ( *(m_gtCaloTemplate->objectParameter()) )[iCondition];

    // check energy threshold
    if ( !checkThreshold(objPar.etThreshold, cand.rank(), m_gtCaloTemplate->condGEq()) ) {
        return false;
    }

    // check eta
    if (!checkBit(objPar.etaRange, cand.etaIndex())) {
        return false;
    }

    // check phi

    if (!checkBit(objPar.phiRange, cand.phiIndex())) {
        return false;
    }

    // particle matches if we get here
    //LogTrace("L1GtCaloCondition")
    //    << "  checkObjectParameter: calorimeter object OK, passes all requirements\n"
    //    << std::endl;

    return true;
}

void L1GtCaloCondition::print(std::ostream& myCout) const {

    m_gtCaloTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

