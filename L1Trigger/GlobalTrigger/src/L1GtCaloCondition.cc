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

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructors
//     default
L1GtCaloCondition::L1GtCaloCondition()
        : L1GtCaloTemplate(), L1GtConditionEvaluation()
{

    m_ifCaloEtaNumberBits = -1;

}

//     from name of the condition
L1GtCaloCondition::L1GtCaloCondition(const std::string& cName)
        : L1GtCaloTemplate(cName), L1GtConditionEvaluation()
{

    m_ifCaloEtaNumberBits = -1;

}

//     from base template condition (from event setup usually)
L1GtCaloCondition::L1GtCaloCondition(const L1GtCaloTemplate& caloTemplate)
        : L1GtCaloTemplate(caloTemplate), L1GtConditionEvaluation()

{

    m_ifCaloEtaNumberBits = -1;

}


// copy constructor
void L1GtCaloCondition::copy(const L1GtCaloCondition &cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_objectParameter = *(cp.objectParameter());
    m_correlationParameter = *(cp.correlationParameter());

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_ifCaloEtaNumberBits = cp.gtIfCaloEtaNumberBits();

}


L1GtCaloCondition::L1GtCaloCondition(const L1GtCaloCondition& cp)
        : L1GtCaloTemplate(cp.m_condName), L1GtConditionEvaluation()
{
    copy(cp);
}

// destructor
L1GtCaloCondition::~L1GtCaloCondition()
{

    // empty

}

// equal operator
L1GtCaloCondition& L1GtCaloCondition::operator= (const L1GtCaloCondition& cp)
{
    copy(cp);
    return *this;
}


// methods

//   set the number of bits for eta of calorimeter objects
void L1GtCaloCondition::setGtIfCaloEtaNumberBits(
    const int& ifCaloEtaNumberBitsValue)
{

    m_ifCaloEtaNumberBits = ifCaloEtaNumberBitsValue;

}


// try all object permutations and check spatial correlations, if required
const bool L1GtCaloCondition::evaluateCondition() const
{


    // number of trigger objects in the condition
    int nObjInCond = nrObjects();


    //LogTrace("L1GtCaloCondition")
    //<< "\n  Maximum number of candidates = " << m_condMaxNumberObjects
    //<< "; objects in condition:  = " << nObjInCond
    //<< std::endl;

    // the candidates
    std::vector<L1GctCand*> candVec(m_condMaxNumberObjects);
    std::vector<int> index(m_condMaxNumberObjects);

    for (int i = 0; i < m_condMaxNumberObjects; ++i) {
        candVec[i] = getCandidate(i);
        index[i] = i;
    }

    int jumpIndex = 1;
    int jump = factorial(m_condMaxNumberObjects - nObjInCond);

    int totalLoops = 0;
    int passLoops = 0;

    // condition result condResult set to true if at least one permutation
    //     passes all requirements
    // all possible permutations are checked
    bool condResult = false;

    // store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the m_combinationsInCond vector
    (*m_combinationsInCond).clear();

    do {

        if (--jumpIndex)
            continue;

        LogTrace("L1GtCaloCondition")
        << "\n  Evaluating new trigger object permutation ... " << std::endl;

        jumpIndex = jump;
        totalLoops++;

        // clear the indices in the combination
        objectsInComb.clear();

        bool tmpResult = true;

        // check if there is a permutation that matches object-parameter requirements
        for (int i = 0; i < nObjInCond; i++) {
            LogTrace("L1GtCaloCondition")
            << "  Current condition index = " << i
            << " < last index = " << nObjInCond
            << ";  checking object with index = " << index[i]
            << std::endl;

            tmpResult &= checkObjectParameter(i, *(candVec)[index[i]] );
            objectsInComb.push_back(index[i]);

        }

        std::ostringstream myCout;
        std::copy(objectsInComb.begin(), objectsInComb.end(),
                  std::ostream_iterator<int> (myCout, " "));

        // if permutation does not match particle conditions
        // skip spatial correlations
        if ( !tmpResult ) {

            LogTrace("L1GtCaloCondition")
            << "  Trigger object permutation ( " <<  myCout.str()
            << ") fails object-parameter requirements."
            << std::endl;

            continue;

        } else {
            LogTrace("L1GtCaloCondition")
            << "  Trigger object permutation ( " <<  myCout.str()
            << ") passes object-parameter requirements."
            << std::endl;
        }


        if (wsc()) {

            LogTrace("L1GtCaloCondition")
            << "\n  Checking spatial correlations: \n"
            << std::endl;

            // wsc requirements have always nObjInCond = 2
            // one can use directly index[0] and index[1] to compute
            // eta and phi differences
            const int ObjInWscComb = 2;
            if (nObjInCond != ObjInWscComb) {
                edm::LogError("L1GtCaloCondition")
                << "\n  Error: "
                << "number of particles in condition with spatial correlation = "
                << nObjInCond << "\n  it must be = " << ObjInWscComb
                << std::endl;
                // TODO Perhaps I should throw here an exception,
                // since something is really wrong if nObjInCond != ObjInWscComb (=2)
                continue;
            }

            unsigned int candDeltaEta;
            unsigned int candDeltaPhi;

            // check candDeltaEta

            // get eta index and the sign bit of the eta index (MSB is the sign)
            //   signedEta[i] is the signed eta index of candVec[index[i]]
            int signedEta[ObjInWscComb];
            int signBit[ObjInWscComb] = {0, 0};

            int scaleEta = 1 << (m_ifCaloEtaNumberBits - 1);
            LogTrace("L1GtCaloCondition")
            << "  scale factor for eta = " << scaleEta << " ( "
            << m_ifCaloEtaNumberBits << " bits for eta of the object, MSB - sign bit), "
            << std::endl;

            for (int i = 0; i < ObjInWscComb; ++i) {
                signBit[i] =
                    (candVec[index[i]]->etaIndex() & scaleEta)>>(m_ifCaloEtaNumberBits - 1);
                signedEta[i] = ( candVec[index[i]]->etaIndex() )%scaleEta;

                if (signBit[i] == 1) {
                    signedEta[i] = (-1)*signedEta[i];
                }

            }

            // compute candDeltaEta - add 1 if signs are different (due to +0/-0 indices)
            candDeltaEta = static_cast<int> (std::abs(signedEta[1] - signedEta[0])) +
                           static_cast<int> (signBit[1]^signBit[0]);

            LogTrace("L1GtCaloCondition")
            << "  candDeltaEta = " << candDeltaEta << " for "
            << "candVec[" << index[0] <<"]->etaIndex() = " << candVec[index[0]]->etaIndex()
            << " (scaled value " << signedEta[0] << "), "
            << "candVec[" << index[1] <<"]->etaIndex() = " << candVec[index[1]]->etaIndex()
            << " (scaled value " << signedEta[1] << "), "
            << std::endl;

            if ( ! checkBit(m_correlationParameter.deltaEtaRange, candDeltaEta) ) {
                LogTrace("L1GtCaloCondition") << "  object deltaEtaRange: failed"
                << std::endl;
                continue;
            } else {
                LogTrace("L1GtCaloCondition")
                << "  ==> object deltaEtaRange: passed"
                << std::endl;
            }

            // check candDeltaPhi

            // calculate absolute value of candDeltaPhi
            if (candVec[index[0]]->phiIndex() > candVec[index[1]]->phiIndex()) {
                candDeltaPhi = candVec[index[0]]->phiIndex() - candVec[index[1]]->phiIndex();
            } else {
                candDeltaPhi = candVec[index[1]]->phiIndex() - candVec[index[0]]->phiIndex();
            }
            LogTrace("L1GtCaloCondition")
            << std::dec << "  candDeltaPhi = " << candDeltaPhi << " for "
            << "candVec[" << index[0] <<"]->phiIndex() = " << candVec[index[0]]->phiIndex()
            << ", "
            << "candVec[" << index[1] <<"]->phiIndex() = " << candVec[index[1]]->phiIndex()
            << std::endl;

            // check if candDeltaPhi > 180 (via delta_phi_maxbits)
            // delta_phi contains bits for 0..180 (0 and 180 included)
            while (candDeltaPhi > m_correlationParameter.deltaPhiMaxbits) {
                LogTrace("L1GtCaloCondition")
                << "  candDeltaPhi = " << candDeltaPhi
                << " > m_correlationParameter.deltaPhiMaxbits ==> needs re-scaling"
                << std::endl;

                // candDeltaPhi > 180 ==> take 360 - candDeltaPhi
                candDeltaPhi = (m_correlationParameter.deltaPhiMaxbits - 1)*2 - candDeltaPhi;
                LogTrace("L1GtCaloCondition")
                << "  candDeltaPhi changed to: " <<  candDeltaPhi
                << std::endl;
            }


            if (!checkBit(m_correlationParameter.deltaPhiRange, candDeltaPhi)) {
                continue;
            }


        } // end wsc check

        // if we get here all checks were successfull for this combination
        // set the general result for evaluateCondition to "true"

        condResult = true;
        passLoops++;
        (*m_combinationsInCond).push_back(objectsInComb);

        LogTrace("L1GtCaloCondition")
        << "\n  ... Trigger object permutation ( " <<  myCout.str()
        << ") passes all requirements."
        << std::endl;


        //    } while ( std::next_permutation(index, index + nObj) );
    } while ( std::next_permutation(index.begin(), index.end()) );


    LogTrace("L1GtCaloCondition")
    << "\n  L1GtCaloCondition: total number of permutations found:          "
    << totalLoops
    << "\n  L1GtCaloCondition: number of permutations passing requirements: "
    << passLoops << "\n"
    << std::endl;

    if ( edm::isDebugEnabled() ) {
        CombinationsInCond::const_iterator itVV;
        std::ostringstream myCout1;

        for(itVV  = (*m_combinationsInCond).begin();
                itVV != (*m_combinationsInCond).end(); itVV++) {

            myCout1 << "( ";

            std::copy((*itVV).begin(), (*itVV).end(),
                      std::ostream_iterator<int> (myCout1, " "));

            myCout1 << "); ";

        }


        LogTrace("L1GtCaloCondition")
        << "\n  List of combinations passing all requirements for this condition: \n  "
        <<  myCout1.str()
        << " \n"
        << std::endl;
    }

    return condResult;

}


/**
 * getCandidate - decides what candidate to get using the particletype
 *
 * 
 * @param indexCand The number of the candidate
 * @return A reference to the candidate
 *
 */

L1GctCand* L1GtCaloCondition::getCandidate (int indexCand) const
{

    //    switch (p_particletype) {
    //        case EG:
    //            return (*m_GT.gtPSB()->getListNoIsoEG())[indexCand];
    //            break;
    //        case IEG:
    //            return (*m_GT.gtPSB()->getListIsoEG())[indexCand];
    //            break;
    //        case CJET:
    //            return (*m_GT.gtPSB()->getListCenJet())[indexCand];
    //            break;
    //        case FJET:
    //            return (*m_GT.gtPSB()->getListForJet())[indexCand];
    //            break;
    //        case TJET:
    //            return (*m_GT.gtPSB()->getListTauJet())[indexCand];
    //            break;
    //        default:
    //            return 0;
    //            break;
    //    }

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

const bool L1GtCaloCondition::checkObjectParameter(
    const int iCondition, const L1GctCand& cand) const
{

    std::string checkFalse = "\n  ==> checkObjectParameter = false ";

    // number of objects in condition
    int nObjInCond = nrObjects();

    if (iCondition >= nObjInCond || iCondition < 0) {
        LogTrace("L1GtCaloCondition")
        << "L1GtCaloCondition:"
        << "  Number of condition outside [0, nObj) interval."
        << "  checkObjectParameter = false "
        << std::endl;
        return false;
    }

    // empty candidates can not be compared
    if (cand.empty()) {
        LogTrace("L1GtCaloCondition")
        << "  Empty calo candidate (" << &cand << ")."
        << "  checkObjectParameter = false "
        << std::endl;
        return false;
    }

    LogTrace("L1GtCaloCondition")
    << "\n  Non-empty calorimeter object: checkObjectParameter starting"
    << std::endl;

    // check energy threshold
    if ( !checkThreshold(
                m_objectParameter[iCondition].etThreshold, cand.rank(), m_condGEq) ) {
        LogTrace("L1GtCaloCondition")
        << "  calo etThreshold: failed"
        << checkFalse
        << std::endl;
        return false;
    }

    // check eta
    LogTrace("L1GtCaloCondition") << "  calo object eta check:"
    << " etaIndex() = "
    << std::dec << cand.etaIndex() << " (dec) "
    << std::hex << cand.etaIndex() << " (hex) "
    << std::dec
    << std::endl;

    if (!checkBit(m_objectParameter[iCondition].etaRange, cand.etaIndex())) {
        LogTrace("L1GtCaloCondition")
        << "  calo object eta: failed"
        << checkFalse
        << std::endl;
        return false;
    } else {
        LogTrace("L1GtCaloCondition")
        << "  ==> calo object eta: passed"
        << std::endl;
    }


    // check phi
    LogTrace("L1GtCaloCondition") << "  calo object phi check:"
    << " phiIndex() = "
    << std::dec << cand.phiIndex() << " (dec) "
    << std::hex << cand.phiIndex() << " (hex) "
    << std::dec
    << std::endl;

    if (!checkBit(m_objectParameter[iCondition].phiRange, cand.phiIndex())) {
        LogTrace("L1GtCaloCondition")
        << "  calo object phi: failed"
        << checkFalse
        << std::endl;
        return false;
    } else {
        LogTrace("L1GtCaloCondition")
        << "  ==> calo object phi: passed"
        << std::endl;
    }

    // particle matches if we get here
    LogTrace("L1GtCaloCondition")
    << "  checkObjectParameter: calorimeter object OK, passes all requirements\n"
    << std::endl;

    return true;
}

void L1GtCaloCondition::print(std::ostream& myCout) const
{

    L1GtCaloTemplate::print(myCout);

    myCout
    << "FIXME evaluation part"
    << std::endl;

}

