/**
 * \class L1uGtCaloCondition
 *
 *
 * Description: evaluation of a CondCalo condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/L1uGtCaloCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1uGtCaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/L1uGtConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
/*#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
*/

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/L1TGlobal/interface/L1uGtBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::L1uGtCaloCondition::L1uGtCaloCondition() :
    L1uGtConditionEvaluation() {

    m_ifCaloEtaNumberBits = -1;
    m_corrParDeltaPhiNrBins = 0;

}

//     from base template condition (from event setup usually)
l1t::L1uGtCaloCondition::L1uGtCaloCondition(const L1uGtCondition* caloTemplate, const L1uGtBoard* ptrGTB,
        const int nrL1EG,
        const int nrL1Jet,
        const int nrL1Tau,
        const int ifCaloEtaNumberBits) :
    L1uGtConditionEvaluation(),
    m_gtCaloTemplate(static_cast<const L1uGtCaloTemplate*>(caloTemplate)),
    m_uGtB(ptrGTB),
    m_ifCaloEtaNumberBits(ifCaloEtaNumberBits)
{

    m_corrParDeltaPhiNrBins = 0;

    // maximum number of  objects received for the evaluation of the condition
    // retrieved before from event setup
    // for a CondCalo, all objects ar of same type, hence it is enough to get the
    // type for the first object

    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            m_condMaxNumberObjects = nrL1EG;
            break;
/*        case IsoEG:
            m_condMaxNumberObjects = nrL1IsoEG;
            break;
*/
        case CenJet:
            m_condMaxNumberObjects = nrL1Jet;
            break;
/*        case ForJet:
            m_condMaxNumberObjects = nrL1ForJet;
            break;
*/
        case TauJet:
            m_condMaxNumberObjects = nrL1Tau;
            break;
        default:
            m_condMaxNumberObjects = 0;
            break;
    }

}

// copy constructor
void l1t::L1uGtCaloCondition::copy(const l1t::L1uGtCaloCondition& cp) {

    m_gtCaloTemplate = cp.gtCaloTemplate();
    m_uGtB = cp.getuGtB();

    m_ifCaloEtaNumberBits = cp.gtIfCaloEtaNumberBits();
    m_corrParDeltaPhiNrBins = cp.m_corrParDeltaPhiNrBins;

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

l1t::L1uGtCaloCondition::L1uGtCaloCondition(const l1t::L1uGtCaloCondition& cp) :
    L1uGtConditionEvaluation() {

    copy(cp);

}

// destructor
l1t::L1uGtCaloCondition::~L1uGtCaloCondition() {

    // empty

}

// equal operator
l1t::L1uGtCaloCondition& l1t::L1uGtCaloCondition::operator=(const l1t::L1uGtCaloCondition& cp) {
    copy(cp);
    return *this;
}

// methods
void l1t::L1uGtCaloCondition::setGtCaloTemplate(const L1uGtCaloTemplate* caloTempl) {

    m_gtCaloTemplate = caloTempl;

}

///   set the pointer to uGT Board
void l1t::L1uGtCaloCondition::setuGtB(const L1uGtBoard* ptrGTB) {

    m_uGtB = ptrGTB;

}

//   set the number of bits for eta of calorimeter objects
void l1t::L1uGtCaloCondition::setGtIfCaloEtaNumberBits(const int& ifCaloEtaNumberBitsValue) {

    m_ifCaloEtaNumberBits = ifCaloEtaNumberBitsValue;

}

//   set the maximum number of bins for the delta phi scales
void l1t::L1uGtCaloCondition::setGtCorrParDeltaPhiNrBins(
        const int& corrParDeltaPhiNrBins) {

    m_corrParDeltaPhiNrBins = corrParDeltaPhiNrBins;

}

// try all object permutations and check spatial correlations, if required
const bool l1t::L1uGtCaloCondition::evaluateCondition(const int bxEval) const {

    // number of trigger objects in the condition
    int nObjInCond = m_gtCaloTemplate->nrObjects();
    //LogTrace("L1GlobalTrigger") << "  nObjInCond: " << nObjInCond
    //    << std::endl;

    // the candidates

    // objectType() gives the type for nrObjects() only,
    // but in a CondCalo all objects have the same type
    // take type from the type of the first object

    const BXVector<const l1t::L1Candidate*>* candVec;

    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            candVec = m_uGtB->getCandL1EG();
            break;

        case CenJet:
            candVec = m_uGtB->getCandL1Jet();
            break;

        case TauJet:
            candVec = m_uGtB->getCandL1Tau();
            break;

        default:
            return false;
            break;
    }

    // Look at objects in bx = bx + relativeBx
    int useBx = bxEval + m_gtCaloTemplate->condRelativeBx();

    // Fail condition if attempting to get Bx outside of range
    if( ( useBx < candVec->getFirstBX() ) ||
	( useBx > candVec->getLastBX() ) ) {
      return false;
    }


    int numberObjects = candVec->size(useBx);
    //LogTrace("L1GlobalTrigger") << "  numberObjects: " << numberObjects
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
    combinationsInCond().clear();

    do {

        if (--jumpIndex)
            continue;

        jumpIndex = jump;
        totalLoops++;

        // clear the indices in the combination
        objectsInComb.clear();

        bool tmpResult = true;

	bool passCondition = false;
        // check if there is a permutation that matches object-parameter requirements
        for (int i = 0; i < nObjInCond; i++) {
	  passCondition = checkObjectParameter(i, *(candVec->at(useBx,index[i]) ));
	  tmpResult &= passCondition;
	  if( passCondition ) 
	    LogDebug("l1t|Global") << "===> L1uGtCaloCondition::evaluateCondition, CONGRATS!! This calo obj passed the condition." << std::endl;
	  else 
	    LogDebug("l1t|Global") << "===> L1uGtCaloCondition::evaluateCondition, FAIL!! This calo obj failed the condition." << std::endl;
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
                    edm::LogError("L1GlobalTrigger")
                        << "\n  Error: "
                        << "number of particles in condition with spatial correlation = "
                        << nObjInCond << "\n  it must be = " << ObjInWscComb
                        << std::endl;
                }

                continue;
            }

            L1uGtCaloTemplate::CorrelationParameter corrPar =
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
                signBit[i] = ((candVec->at(useBx,index[i]))->hwEta() & scaleEta)
                    >>(m_ifCaloEtaNumberBits - 1);
                signedEta[i] = ((candVec->at(useBx,index[i]))->hwEta() )%scaleEta;

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
            if ((candVec->at(useBx,index[0]))->hwPhi()> (candVec->at(useBx,index[1]))->hwPhi()) {
                candDeltaPhi = (candVec->at(useBx,index[0]))->hwPhi() - (candVec->at(useBx,index[1]))->hwPhi();
            }
            else {
                candDeltaPhi = (candVec->at(useBx,index[1]))->hwPhi() - (candVec->at(useBx,index[0]))->hwPhi();
            }

            // check if candDeltaPhi > 180 (via delta_phi_maxbits)
            // delta_phi contains bits for 0..180 (0 and 180 included)
            // protect also against infinite loop...

            int nMaxLoop = 10;
            int iLoop = 0;

            while (candDeltaPhi >= m_corrParDeltaPhiNrBins) {

                unsigned int candDeltaPhiInitial = candDeltaPhi;

                // candDeltaPhi > 180 ==> take 360 - candDeltaPhi
                candDeltaPhi = (m_corrParDeltaPhiNrBins - 1) * 2 - candDeltaPhi;
                if (m_verbosity) {
                    LogTrace("L1GlobalTrigger")
                            << "    Initial candDeltaPhi = "
                            << candDeltaPhiInitial
                            << " > m_corrParDeltaPhiNrBins = "
                            << m_corrParDeltaPhiNrBins
                            << "  ==> candDeltaPhi rescaled to: "
                            << candDeltaPhi << " [ loop index " << iLoop
                            << "; breaks after " << nMaxLoop << " loops ]\n"
                            << std::endl;
                }

                iLoop++;
                if (iLoop > nMaxLoop) {
                    return false;
                }
            }


            if (!checkBit(corrPar.deltaPhiRange, candDeltaPhi)) {
                continue;
            }

        } // end wsc check

        // if we get here all checks were successful for this combination
        // set the general result for evaluateCondition to "true"

        condResult = true;
        passLoops++;
        combinationsInCond().push_back(objectsInComb);

        //    } while ( std::next_permutation(index, index + nObj) );
    } while (std::next_permutation(index.begin(), index.end()) );

    //LogTrace("L1GlobalTrigger")
    //    << "\n  L1uGtCaloCondition: total number of permutations found:          " << totalLoops
    //    << "\n  L1uGtCaloCondition: number of permutations passing requirements: " << passLoops
    //    << "\n" << std::endl;

    return condResult;

}

// load calo candidates
const l1t::L1Candidate* l1t::L1uGtCaloCondition::getCandidate(const int bx, const int indexCand) const {

    // objectType() gives the type for nrObjects() only,
    // but in a CondCalo all objects have the same type
    // take type from the type of the first object
    switch ((m_gtCaloTemplate->objectType())[0]) {
        case NoIsoEG:
            return (m_uGtB->getCandL1EG())->at(bx,indexCand);
            break;

        case CenJet:
            return (m_uGtB->getCandL1Jet())->at(bx,indexCand);
            break;

       case TauJet:
            return (m_uGtB->getCandL1Tau())->at(bx,indexCand);
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

const bool l1t::L1uGtCaloCondition::checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const {

    // number of objects in condition
    int nObjInCond = m_gtCaloTemplate->nrObjects();

    if (iCondition >= nObjInCond || iCondition < 0) {
        return false;
    }

    // empty candidates can not be compared
//     if (cand.empty()) {
//         return false;
//     }

    const L1uGtCaloTemplate::ObjectParameter objPar = ( *(m_gtCaloTemplate->objectParameter()) )[iCondition];

    LogDebug("l1t|Global")
      << "\n L1uGtCaloTemplate: "
      << "\n\t condRelativeBx = " << m_gtCaloTemplate->condRelativeBx()
      << "\n ObjectParameter : "
      << "\n\t etThreshold = " << objPar.etThreshold
      << "\n\t etaRange    = " << objPar.etaRange
      << "\n\t phiRange    = " << objPar.phiRange
      << std::endl;

    LogDebug("l1t|Global")
      << "\n l1t::Candidate : "
      << "\n\t hwPt   = " <<  cand.hwPt()
      << "\n\t hwEta  = " << cand.hwEta()
      << "\n\t hwPhi  = " << cand.hwPhi()
      << std::endl;


    // check energy threshold
    if ( !checkThreshold(objPar.etThreshold, cand.hwPt(), m_gtCaloTemplate->condGEq()) ) {
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkThreshold" << std::endl;
        return false;
    }

    // check eta
    if( !checkRange(cand.hwEta(), objPar.etaWindowBegin, objPar.etaWindowEnd, objPar.etaWindowVetoBegin, objPar.etaWindowVetoEnd) ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRange(eta)" << std::endl;
      return false;
    }

//     if (!checkBit(objPar.etaRange, cand.hwEta())) {
//         return false;
//     }

    // check phi
    if( !checkRange(cand.hwPhi(), objPar.phiWindowBegin, objPar.phiWindowEnd, objPar.phiWindowVetoBegin, objPar.phiWindowVetoEnd) ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRange(phi)" << std::endl;
      return false;
    }

//     if (!checkBit(objPar.phiRange, cand.hwPhi())) {
//         return false;
//     }

    // particle matches if we get here
    //LogTrace("L1GlobalTrigger")
    //    << "  checkObjectParameter: calorimeter object OK, passes all requirements\n"
    //    << std::endl;

    return true;
}

void l1t::L1uGtCaloCondition::print(std::ostream& myCout) const {

    m_gtCaloTemplate->print(myCout);

    myCout << "    Number of bits for eta of calorimeter objects = "
            << m_ifCaloEtaNumberBits << std::endl;
    myCout << "    Maximum number of bins for the delta phi scales = "
            << m_corrParDeltaPhiNrBins << "\n " << std::endl;

    L1uGtConditionEvaluation::print(myCout);

}

