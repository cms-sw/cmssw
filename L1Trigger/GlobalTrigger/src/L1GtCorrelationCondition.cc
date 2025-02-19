/**
 * \class L1GtCorrelationCondition
 *
 *
 * Description: evaluation of a CondCorrelation condition.
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
#include "L1Trigger/GlobalTrigger/interface/L1GtCorrelationCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtMuonCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCaloCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtEnergySumCondition.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtEtaPhiConversions.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
L1GtCorrelationCondition::L1GtCorrelationCondition() :
    L1GtConditionEvaluation(),
    m_isDebugEnabled(edm::isDebugEnabled())
{

    // empty

}

//     from base template condition (from event setup usually)
L1GtCorrelationCondition::L1GtCorrelationCondition(
        const L1GtCondition* corrTemplate, const L1GtCondition* cond0Condition,
        const L1GtCondition* cond1Condition, const int cond0NrL1Objects,
        const int cond1NrL1Objects, const int cond0EtaBits,
        const int cond1EtaBits, const L1GlobalTriggerGTL* ptrGTL,
        const L1GlobalTriggerPSB* ptrPSB,
        const L1GtEtaPhiConversions* etaPhiConversions) :
            L1GtConditionEvaluation(),
            m_gtCorrelationTemplate(static_cast<const L1GtCorrelationTemplate*>(corrTemplate)),
            m_gtCond0(cond0Condition), m_gtCond1(cond1Condition),
            m_cond0NrL1Objects(cond0NrL1Objects),
            m_cond1NrL1Objects(cond1NrL1Objects), m_cond0EtaBits(cond0EtaBits),
            m_cond1EtaBits(cond1EtaBits), m_gtGTL(ptrGTL), m_gtPSB(ptrPSB),
            m_gtEtaPhiConversions(etaPhiConversions),
            m_isDebugEnabled(edm::isDebugEnabled())
            {

    m_condMaxNumberObjects = 2; // irrelevant for correlation conditions
    m_nrBinsPhi = 0;
}

// copy constructor
void L1GtCorrelationCondition::copy(const L1GtCorrelationCondition &cp) {

    m_gtCorrelationTemplate = cp.m_gtCorrelationTemplate;

    m_gtCond0 = cp.m_gtCond0;
    m_gtCond1 = cp.m_gtCond1;

    m_cond0NrL1Objects = cp.m_cond0NrL1Objects;
    m_cond1NrL1Objects = cp.m_cond1NrL1Objects;
    m_cond0EtaBits = cp.m_cond0EtaBits;
    m_cond1EtaBits = cp.m_cond1EtaBits;

    m_nrBinsPhi = cp.m_nrBinsPhi;

    m_gtCorrelationTemplate = cp.m_gtCorrelationTemplate;
    m_gtGTL = cp.m_gtGTL;
    m_gtPSB = cp.m_gtPSB;

    m_gtEtaPhiConversions = cp.m_gtEtaPhiConversions;

    m_condMaxNumberObjects = cp.m_condMaxNumberObjects;
    m_condLastResult = cp.m_condLastResult;
    m_combinationsInCond = cp.m_combinationsInCond;

    m_verbosity = cp.m_verbosity;
    m_isDebugEnabled = cp.m_isDebugEnabled;


}

L1GtCorrelationCondition::L1GtCorrelationCondition(
        const L1GtCorrelationCondition& cp) :
    L1GtConditionEvaluation() {
    copy(cp);
}

// destructor
L1GtCorrelationCondition::~L1GtCorrelationCondition() {

    // empty

}

// equal operator
L1GtCorrelationCondition& L1GtCorrelationCondition::operator= (
        const L1GtCorrelationCondition& cp)
{
    copy(cp);
    return *this;
}

// methods

void L1GtCorrelationCondition::setGtNrBinsPhi(const unsigned int nrBins) {

    m_nrBinsPhi = nrBins;

}

//
void L1GtCorrelationCondition::setGtCorrelationTemplate(
        const L1GtCorrelationTemplate* corrTempl) {

    m_gtCorrelationTemplate = corrTempl;

}

//   set the pointer to GTL
void L1GtCorrelationCondition::setGtGTL(const L1GlobalTriggerGTL* ptrGTL) {

    m_gtGTL = ptrGTL;

}

//   set the pointer to PSB
void L1GtCorrelationCondition::setGtPSB(const L1GlobalTriggerPSB* ptrPSB) {

    m_gtPSB = ptrPSB;

}

// try all object permutations and check spatial correlations, if required
const bool L1GtCorrelationCondition::evaluateCondition() const {

    // std::cout << "m_isDebugEnabled = " << m_isDebugEnabled << std::endl;
    // std::cout << "m_verbosity = " << m_verbosity << std::endl;

    bool condResult = false;
    bool reqObjResult = false;

    // number of objects in condition (it is 2, no need to retrieve from
    // condition template) and their type
    int nObjInCond = 2;
    std::vector<L1GtObject> cndObjTypeVec(nObjInCond);

    // evaluate first the two sub-conditions (Type1s)

    const L1GtConditionCategory cond0Categ = m_gtCorrelationTemplate->cond0Category();
    const L1GtConditionCategory cond1Categ = m_gtCorrelationTemplate->cond1Category();

    const L1GtMuonTemplate* corrMuon = 0;
    const L1GtCaloTemplate* corrCalo = 0;
    const L1GtEnergySumTemplate* corrEnergySum = 0;

    // FIXME copying is slow...
    CombinationsInCond cond0Comb;
    CombinationsInCond cond1Comb;

    switch (cond0Categ) {
        case CondMuon: {
            corrMuon = static_cast<const L1GtMuonTemplate*>(m_gtCond0);
            L1GtMuonCondition muCondition(corrMuon, m_gtGTL,
                    m_cond0NrL1Objects, m_cond0EtaBits);

            muCondition.evaluateConditionStoreResult();
            reqObjResult = muCondition.condLastResult();

            cond0Comb = (muCondition.getCombinationsInCond());
            cndObjTypeVec[0] = (corrMuon->objectType())[0];

            if (m_verbosity && m_isDebugEnabled ) {
                std::ostringstream myCout;
                muCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }
        }
            break;
        case CondCalo: {
            corrCalo = static_cast<const L1GtCaloTemplate*>(m_gtCond0);

            L1GtCaloCondition caloCondition(corrCalo, m_gtPSB,
                    m_cond0NrL1Objects, m_cond0NrL1Objects, m_cond0NrL1Objects,
                    m_cond0NrL1Objects, m_cond0NrL1Objects, m_cond0EtaBits);

            caloCondition.evaluateConditionStoreResult();
            reqObjResult = caloCondition.condLastResult();

            cond0Comb = (caloCondition.getCombinationsInCond());
            cndObjTypeVec[0] = (corrCalo->objectType())[0];

            if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCout;
                caloCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }
        }
            break;
        case CondEnergySum: {
            corrEnergySum = static_cast<const L1GtEnergySumTemplate*>(m_gtCond0);
            L1GtEnergySumCondition eSumCondition(corrEnergySum, m_gtPSB);

            eSumCondition.evaluateConditionStoreResult();
            reqObjResult = eSumCondition.condLastResult();

            cond0Comb = (eSumCondition.getCombinationsInCond());
            cndObjTypeVec[0] = (corrEnergySum->objectType())[0];

            if (m_verbosity && m_isDebugEnabled ) {
                std::ostringstream myCout;
                eSumCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }
        }
            break;
        default: {
            // should not arrive here, there are no correlation conditions defined for this object
            return false;
        }
            break;
    }

    // return if first subcondition is false
    if (!reqObjResult) {
        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GlobalTrigger")
                    << "\n  First sub-condition false, second sub-condition not evaluated and not printed."
                    << std::endl;
        }
        return false;
    }

    // second object
    reqObjResult = false;

    switch (cond1Categ) {
        case CondMuon: {
            corrMuon = static_cast<const L1GtMuonTemplate*>(m_gtCond1);
            L1GtMuonCondition muCondition(corrMuon, m_gtGTL,
                    m_cond1NrL1Objects, m_cond1EtaBits);

            muCondition.evaluateConditionStoreResult();
            reqObjResult = muCondition.condLastResult();

            cond1Comb = (muCondition.getCombinationsInCond());
            cndObjTypeVec[1] = (corrMuon->objectType())[0];

            if (m_verbosity && m_isDebugEnabled ) {
                std::ostringstream myCout;
                muCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }
        }
            break;
        case CondCalo: {
            corrCalo = static_cast<const L1GtCaloTemplate*>(m_gtCond1);
            L1GtCaloCondition caloCondition(corrCalo, m_gtPSB,
                    m_cond1NrL1Objects, m_cond1NrL1Objects, m_cond1NrL1Objects,
                    m_cond1NrL1Objects, m_cond1NrL1Objects, m_cond1EtaBits);

            caloCondition.evaluateConditionStoreResult();
            reqObjResult = caloCondition.condLastResult();

            cond1Comb = (caloCondition.getCombinationsInCond());
            cndObjTypeVec[1] = (corrCalo->objectType())[0];

            if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCout;
                caloCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }

        }
            break;
        case CondEnergySum: {
            corrEnergySum = static_cast<const L1GtEnergySumTemplate*>(m_gtCond1);
            L1GtEnergySumCondition eSumCondition(corrEnergySum, m_gtPSB);
            eSumCondition.evaluateConditionStoreResult();
            reqObjResult = eSumCondition.condLastResult();

            cond1Comb = (eSumCondition.getCombinationsInCond());
            cndObjTypeVec[1] = (corrEnergySum->objectType())[0];

            if (m_verbosity && m_isDebugEnabled ) {
                std::ostringstream myCout;
                eSumCondition.print(myCout);

                LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
            }
        }
            break;
        default: {
            // should not arrive here, there are no correlation conditions defined for this object
            return false;
        }
            break;
    }

    // return if second sub-condition is false
    if (!reqObjResult) {
        return false;
    } else {
        LogTrace("L1GlobalTrigger") << "\n"
                << "    Both sub-conditions true for object requirements."
                << "    Evaluate correlation requirements.\n" << std::endl;

    }

    //
    // evaluate the delta_eta and delta_phi correlations
    // if here, the object requirements are satisfied for both sub-conditions
    //

    // get the correlation parameters

    L1GtCorrelationTemplate::CorrelationParameter corrPar =
        *(m_gtCorrelationTemplate->correlationParameter());

    // convert the eta template requirements from string to 64-bit integers
    // number of 64-bit integers: string length / 16
    size_t deltaEtaRangeConvSize = (corrPar.deltaEtaRange).size() / 16
            + 1;
    std::vector<unsigned long long> deltaEtaRangeConv(
            deltaEtaRangeConvSize);

    if (!(hexStringToInt64(corrPar.deltaEtaRange, deltaEtaRangeConv))) {
        return false;
    }

    // convert the phi template requirements from string to 64-bit integers
    // number of 64-bit integers: string length / 16
    size_t deltaPhiRangeConvSize = (corrPar.deltaPhiRange).size() / 16
            + 1;
    std::vector<unsigned long long> deltaPhiRangeConv(
            deltaPhiRangeConvSize);

    if (!(hexStringToInt64(corrPar.deltaPhiRange, deltaPhiRangeConv))) {
        return false;
    }

    // get the index of L1 GT object pair
    unsigned int objPairIndex = (m_gtEtaPhiConversions->gtObjectPairIndex(
            cndObjTypeVec[0], cndObjTypeVec[1]));

    // get the maximum number of bins for the delta phi scales
    unsigned int corrParDeltaPhiNrBins =
            (m_gtEtaPhiConversions->gtObjectNrBinsPhi(objPairIndex)) / 2 + 1;


    // vector to store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;
    objectsInComb.reserve(nObjInCond);

    // clear the m_combinationsInCond vector
    (combinationsInCond()).clear();

    // pointers to objects
    const std::vector<const L1MuGMTCand*>* candMuVec = 0;
    const std::vector<const L1GctCand*>* candCaloVec = 0;
    //    only ETM and HTM  can appear in correlation conditions
    const L1GctEtMiss* candETM = 0;
    const L1GctHtMiss* candHTM = 0;

    // make the conversions of the indices, depending on the combination of objects involved
    // (via pair index)

    unsigned int phiIndex0 = 0;
    unsigned int phiIndex1 = 0;

    unsigned int phiIndex0Converted = 0;
    unsigned int phiIndex1Converted = 0;

    unsigned int etaIndex0 = 0;
    unsigned int etaIndex1 = 0;

    unsigned int etaIndex0Converted = 0;
    unsigned int etaIndex1Converted = 0;

    LogTrace("L1GlobalTrigger")
            << "  Sub-condition 0: std::vector<SingleCombInCond> size: "
            << (cond0Comb.size()) << std::endl;
    LogTrace("L1GlobalTrigger")
            << "  Sub-condition 1: std::vector<SingleCombInCond> size: "
            << (cond1Comb.size()) << std::endl;


    // loop over all combinations which produced individually "true" as Type1s
    for (std::vector<SingleCombInCond>::const_iterator it0Comb =
            cond0Comb.begin(); it0Comb != cond0Comb.end(); it0Comb++) {

        // Type1s: there is 1 object only, no need for a loop, index 0 should be OK in (*it0Comb)[0]
        // ... but add protection to not crash
        int obj0Index = -1;

        if ((*it0Comb).size() > 0) {
            obj0Index = (*it0Comb)[0];
        } else {
            LogTrace("L1GlobalTrigger")
                    << "\n  SingleCombInCond (*it0Comb).size() "
                    << ((*it0Comb).size()) << std::endl;
            return false;
        }

        bool convStatus = false;

        switch (cond0Categ) {
            case CondMuon: {
                candMuVec = m_gtGTL->getCandL1Mu();
                phiIndex0 = (*candMuVec)[obj0Index]->phiIndex();
                etaIndex0 = (*candMuVec)[obj0Index]->etaIndex();

                convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                        objPairIndex,  0, phiIndex0, phiIndex0Converted);

                if (!convStatus) {
                    // conversion failed, message written in L1GtEtaPhiConversions
                    return false;
                }

                convStatus = m_gtEtaPhiConversions->convertEtaIndex(
                        Mu, etaIndex0, etaIndex0Converted);

                if (!convStatus) {
                    // conversion failed, message written in L1GtEtaPhiConversions
                    return false;
                }

            }
                break;
            case CondCalo: {
                switch (cndObjTypeVec[0]) {
                    case NoIsoEG: {
                        candCaloVec = m_gtPSB->getCandL1NoIsoEG();
                    }
                        break;
                    case IsoEG: {
                        candCaloVec = m_gtPSB->getCandL1IsoEG();
                    }
                        break;
                    case CenJet: {
                        candCaloVec = m_gtPSB->getCandL1CenJet();
                    }
                        break;
                    case ForJet: {
                        candCaloVec = m_gtPSB->getCandL1ForJet();
                    }
                        break;
                    case TauJet: {
                        candCaloVec = m_gtPSB->getCandL1TauJet();
                    }
                        break;
                    default: {
                        // do nothing
                    }
                        break;
                }

                phiIndex0 = (*candCaloVec)[obj0Index]->phiIndex();
                etaIndex0 = (*candCaloVec)[obj0Index]->etaIndex();

                convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                        objPairIndex,  0, phiIndex0, phiIndex0Converted);

                if (!convStatus) {
                    // conversion failed, message written in L1GtEtaPhiConversions
                    return false;
                }

                convStatus = m_gtEtaPhiConversions->convertEtaIndex(
                        cndObjTypeVec[0], etaIndex0, etaIndex0Converted);

                if (!convStatus) {
                    // conversion failed, message written in L1GtEtaPhiConversions
                    return false;
                }
            }
                break;
            case CondEnergySum: {
                switch (cndObjTypeVec[0]) {
                    case ETM: {
                        candETM = m_gtPSB->getCandL1ETM();
                        phiIndex0 = candETM->phi();

                        convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                                objPairIndex,  0, phiIndex0, phiIndex0Converted);

                        if (!convStatus) {
                            // conversion failed, message written in L1GtEtaPhiConversions
                            return false;
                        }

                    }
                        break;
                    case HTM: {
                        candHTM = m_gtPSB->getCandL1HTM();
                        phiIndex0 = candHTM->phi();

                        convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                                objPairIndex,  0, phiIndex0, phiIndex0Converted);

                        if (!convStatus) {
                            // conversion failed, message written in L1GtEtaPhiConversions
                            return false;
                        }

                    }
                        break;
                    default:
                        // do nothing
                        break;
                }
            }
                break;
            default: {
                // should not arrive here, there are no correlation conditions defined for this object
                return false;
            }
                break;
        }

        for (std::vector<SingleCombInCond>::const_iterator it1Comb =
                cond1Comb.begin(); it1Comb != cond1Comb.end(); it1Comb++) {

            // Type1s: there is 1 object only, no need for a loop (*it1Comb)[0]
            // ... but add protection to not crash
            int obj1Index = -1;

            if ((*it1Comb).size() > 0) {
                obj1Index = (*it1Comb)[0];
            } else {
                LogTrace("L1GlobalTrigger")
                        << "\n  SingleCombInCond (*it1Comb).size() "
                        << ((*it1Comb).size()) << std::endl;
                return false;
            }

            switch (cond1Categ) {
                case CondMuon: {
                    candMuVec = m_gtGTL->getCandL1Mu();
                    phiIndex1 = (*candMuVec)[obj1Index]->phiIndex();
                    etaIndex1 = (*candMuVec)[obj1Index]->etaIndex();

                    convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                            objPairIndex,  1, phiIndex1, phiIndex1Converted);

                    if (!convStatus) {
                        // conversion failed, message written in L1GtEtaPhiConversions
                        return false;
                    }

                    convStatus = m_gtEtaPhiConversions->convertEtaIndex(
                            cndObjTypeVec[1], etaIndex1, etaIndex1Converted);

                    if (!convStatus) {
                        // conversion failed, message written in L1GtEtaPhiConversions
                        return false;
                    }
                }
                    break;
                case CondCalo: {
                    switch (cndObjTypeVec[1]) {
                        case NoIsoEG:
                            candCaloVec = m_gtPSB->getCandL1NoIsoEG();
                            break;
                        case IsoEG:
                            candCaloVec = m_gtPSB->getCandL1IsoEG();
                            break;
                        case CenJet:
                            candCaloVec = m_gtPSB->getCandL1CenJet();
                            break;
                        case ForJet:
                            candCaloVec = m_gtPSB->getCandL1ForJet();
                            break;
                        case TauJet:
                            candCaloVec = m_gtPSB->getCandL1TauJet();
                            break;
                        default:
                            // do nothing
                            break;
                    }

                    phiIndex1 = (*candCaloVec)[obj1Index]->phiIndex();
                    etaIndex1 = (*candCaloVec)[obj1Index]->etaIndex();

                    convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                            objPairIndex,  1, phiIndex1, phiIndex1Converted);

                    if (!convStatus) {
                        // conversion failed, message written in L1GtEtaPhiConversions
                        return false;
                    }

                    convStatus = m_gtEtaPhiConversions->convertEtaIndex(
                            cndObjTypeVec[1], etaIndex1, etaIndex1Converted);

                    if (!convStatus) {
                        // conversion failed, message written in L1GtEtaPhiConversions
                        return false;
                    }

                }
                    break;
                case CondEnergySum: {
                    switch (cndObjTypeVec[1]) {
                        case ETM: {
                            candETM = m_gtPSB->getCandL1ETM();
                            phiIndex1 = candETM->phi();

                            convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                                    objPairIndex,  1, phiIndex1, phiIndex1Converted);

                            if (!convStatus) {
                                // conversion failed, message written in L1GtEtaPhiConversions
                                return false;
                            }
                        }
                            break;
                        case HTM: {
                            candHTM = m_gtPSB->getCandL1HTM();
                            phiIndex1 = candHTM->phi();

                            convStatus = m_gtEtaPhiConversions->convertPhiIndex(
                                    objPairIndex,  1, phiIndex1, phiIndex1Converted);

                            if (!convStatus) {
                                // conversion failed, message written in L1GtEtaPhiConversions
                                return false;
                            }
                        }
                            break;
                        default:
                            // do nothing
                            break;
                    }
                }
                    break;
                default: {
                    // should not arrive here, there are no correlation conditions defined for this object
                    return false;
                }
                    break;
            }

            if (m_verbosity && m_isDebugEnabled ) {
                LogTrace("L1GlobalTrigger") << "    Correlation pair ["
                        << l1GtObjectEnumToString(cndObjTypeVec[0]) << ", "
                        << l1GtObjectEnumToString(cndObjTypeVec[1])
                        << "] with collection indices [" << obj0Index << ", "
                        << obj1Index << "] " << " has: \n      phi indices = ["
                        << phiIndex0 << ", " << phiIndex1 << "] converted to ["
                        << phiIndex0Converted << ", " << phiIndex1Converted
                        << "] \n      eta indices [" << etaIndex0 << ", "
                        << etaIndex1 << "] converted to ["
                        << etaIndex0Converted << ", " << etaIndex1Converted
                        << "] \n" << std::endl;
            }

            // the conversions were successful, need to evaluate requirements now

            bool reqEtaPhiResult = false;

            // evaluate candDeltaPhi requirements

            unsigned int candDeltaPhi;

            // calculate absolute value of candDeltaPhi
            if (phiIndex0Converted > phiIndex1Converted) {
                candDeltaPhi = phiIndex0Converted - phiIndex1Converted;
            } else {
                candDeltaPhi = phiIndex1Converted - phiIndex0Converted;
            }

            // check if candDeltaPhi > 180 (via delta_phi_maxbits)
            // delta_phi contains bits for 0..180 (0 and 180 included)
            // protect also against infinite loop...

            int nMaxLoop = 10;
            int iLoop = 0;

            while (candDeltaPhi >= corrParDeltaPhiNrBins) {

                unsigned int candDeltaPhiInitial = candDeltaPhi;

                // candDeltaPhi > 180 ==> take 360 - candDeltaPhi
                candDeltaPhi = (corrParDeltaPhiNrBins - 1) * 2 - candDeltaPhi;
                if (m_verbosity) {
                    LogTrace("L1GlobalTrigger")
                            << "    Initial candDeltaPhi = "
                            << candDeltaPhiInitial
                            << " > corrParDeltaPhiNrBins = "
                            << corrParDeltaPhiNrBins
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


            // template requirements already converted from string to 64-bit integers
            // ...now check for each 64-bit integer against template requirements
            bool indResult = true;

            for (size_t iDeltaPhi = 0; iDeltaPhi < deltaPhiRangeConv.size(); ++iDeltaPhi) {
                if (!checkBit(deltaPhiRangeConv[iDeltaPhi], candDeltaPhi)) {
                    indResult = false;
                }
            }

            if (!indResult) {

                if (m_verbosity && m_isDebugEnabled ) {
                    LogTrace("L1GlobalTrigger") << "      object delta phi = "
                            << candDeltaPhi << " fails delta phi requirements."
                            << "\n      Pair fails correlation condition.\n"
                            << std::endl;
                }

                reqEtaPhiResult = false;
                continue;

            } else {
                if (m_verbosity && m_isDebugEnabled ) {
                    LogTrace("L1GlobalTrigger") << "      object delta phi = "
                            << candDeltaPhi
                            << " passes delta phi requirements."
                            << std::endl;
                }
            }


            // evaluate candDeltaEta requirements

            unsigned int candDeltaEta;

            // calculate absolute value of candDeltaEta
            if (etaIndex0Converted > etaIndex1Converted) {
                candDeltaEta = etaIndex0Converted - etaIndex1Converted;
            } else {
                candDeltaEta = etaIndex1Converted - etaIndex0Converted;
            }

            // template requirements already converted from string to 64-bit integers
            // ...now check for each 64-bit integer against template requirements
            indResult = true;

            for (size_t iDeltaEta = 0; iDeltaEta < deltaEtaRangeConv.size(); ++iDeltaEta) {
                if (!checkBit(deltaEtaRangeConv[iDeltaEta], candDeltaEta)) {
                    indResult = false;
                }
            }

            if (!indResult) {

                if (m_verbosity && m_isDebugEnabled ) {
                    LogTrace("L1GlobalTrigger") << "      object delta eta = "
                            << candDeltaEta << " fails delta eta requirements."
                            << "\n      Pair fails correlation condition.\n"
                            << std::endl;
                }

                reqEtaPhiResult = false;
                continue;

            } else {
                if (m_verbosity && m_isDebugEnabled ) {
                    LogTrace("L1GlobalTrigger") << "      object delta eta = "
                            << candDeltaEta
                            << " passes delta eta requirements."
                            << "\n      Pair passes correlation condition.\n"
                            << std::endl;
                }

                reqEtaPhiResult = true;

            }


            // clear the indices in the combination
            objectsInComb.clear();

            objectsInComb.push_back(obj0Index);
            objectsInComb.push_back(obj1Index);

            // if we get here all checks were successful for this combination
            // set the general result for evaluateCondition to "true"

            if (reqEtaPhiResult) {

                condResult = true;
                (combinationsInCond()).push_back(objectsInComb);

            }

        }

    }

    if (m_verbosity && m_isDebugEnabled && condResult) {
        LogTrace("L1GlobalTrigger") << (combinationsInCond()).size()
                << " correlation pair(s) [" << l1GtObjectEnumToString(
                cndObjTypeVec[0]) << ", " << l1GtObjectEnumToString(
                cndObjTypeVec[1]) << "] pass(es) the correlation condition.\n"
                << std::endl;
    }

    return condResult;

}


void L1GtCorrelationCondition::print(std::ostream& myCout) const {

    m_gtCorrelationTemplate->print(myCout);
    L1GtConditionEvaluation::print(myCout);

}

