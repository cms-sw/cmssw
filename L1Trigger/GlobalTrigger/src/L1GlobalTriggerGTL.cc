/**
 * \class L1GlobalTriggerGTL
 * 
 * 
 * Description: Global Trigger Logic board, see header file for details.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

// system include files
#include <ext/hash_map>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtMuonCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCaloCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtEnergySumCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtJetCountsCondition.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GtCorrelationCondition.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations

// constructor
L1GlobalTriggerGTL::L1GlobalTriggerGTL() :
    m_candL1Mu( new std::vector<const L1MuGMTCand*>) {

    m_gtlAlgorithmOR.reset();
    m_gtlDecisionWord.reset();

    // initialize cached IDs
    m_l1GtMenuCacheID = 0ULL;
}

// destructor
L1GlobalTriggerGTL::~L1GlobalTriggerGTL() {

    reset();
    delete m_candL1Mu;

}

// operations
void L1GlobalTriggerGTL::init(const int nrL1Mu, const int numberPhysTriggers) {

    m_candL1Mu->reserve(nrL1Mu);

    // FIXME move from bitset to std::vector<bool> to be able to use 
    // numberPhysTriggers from EventSetup
    
    //m_gtlAlgorithmOR.reserve(numberPhysTriggers);
    //m_gtlAlgorithmOR.assign(numberPhysTriggers, false);

    //m_gtlDecisionWord.reserve(numberPhysTriggers);
    //m_gtlDecisionWord.assign(numberPhysTriggers, false);

}

// receive data from Global Muon Trigger
void L1GlobalTriggerGTL::receiveGmtObjectData(edm::Event& iEvent,
    const edm::InputTag& muGmtInputTag, const int iBxInEvent, const bool receiveMu,
    const int nrL1Mu) {

    LogDebug("L1GlobalTriggerGTL")
            << "\n**** L1GlobalTriggerGTL receiving muon data for BxInEvent = "
            << iBxInEvent << "\n     from input tag " << muGmtInputTag << "\n"
            << std::endl;

    reset();

    // get data from Global Muon Trigger
    if (receiveMu) {

        edm::Handle<std::vector<L1MuGMTCand> > muonData;
        iEvent.getByLabel(muGmtInputTag, muonData);

        if (!muonData.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: std::vector<L1MuGMTCand> with input tag " << muGmtInputTag
            << "\nrequested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        std::vector< L1MuGMTCand>::const_iterator itMuon;
        for (itMuon = muonData->begin(); itMuon != muonData->end(); itMuon++) {
            if ((*itMuon).bx() == iBxInEvent) {

                (*m_candL1Mu).push_back(&(*itMuon));
                //LogTrace("L1GlobalTriggerGTL") << (*itMuon)
                //        << std::endl;

            }

        }

    }

    if (edm::isDebugEnabled() ) {
        printGmtData(iBxInEvent);
    }

}

// run GTL
void L1GlobalTriggerGTL::run(edm::Event& iEvent, const edm::EventSetup& evSetup,
    const L1GlobalTriggerPSB* ptrGtPSB, const bool produceL1GtObjectMapRecord,
    const int iBxInEvent, std::auto_ptr<L1GlobalTriggerObjectMapRecord>& gtObjectMapRecord,
    const unsigned int numberPhysTriggers,
    const int nrL1Mu,
    const int nrL1NoIsoEG,
    const int nrL1IsoEG,
    const int nrL1CenJet,
    const int nrL1ForJet,
    const int nrL1TauJet,
    const int nrL1JetCounts,
    const int ifMuEtaNumberBits,
    const int ifCaloEtaNumberBits) {


	// get / update the trigger menu from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtMenuCacheID = evSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
        
        edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
        evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;
        m_l1GtMenu =  l1GtMenu.product();
        
        m_l1GtMenuCacheID = l1GtMenuCacheID;
        
    }

    const std::vector<ConditionMap>& conditionMap = m_l1GtMenu->gtConditionMap();
    const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // save the results in temporary maps

    
    std::vector<L1GtAlgorithmEvaluation::ConditionEvaluationMap> conditionResultMaps;
    conditionResultMaps.reserve(conditionMap.size());
    
    for (std::vector<ConditionMap>::const_iterator 
    		itCondOnChip = conditionMap.begin(); itCondOnChip != conditionMap.end(); itCondOnChip++) {

        //L1GtAlgorithmEvaluation::ConditionEvaluationMap cMapResults;
        L1GtAlgorithmEvaluation::ConditionEvaluationMap cMapResults((*itCondOnChip).size()); // hash map
        
        for (CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            // evaluate condition
            switch ((itCond->second)->condCategory()) {
                case CondMuon: {

                    L1GtMuonCondition* muCondition = new L1GtMuonCondition(itCond->second, this,
                            nrL1Mu, ifMuEtaNumberBits);
                    muCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = muCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        muCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << myCout.str() << std::endl;
                    }

                    //delete muCondition;

                }
                    break;
                case CondCalo: {

                    L1GtCaloCondition* caloCondition = new L1GtCaloCondition(
                            itCond->second, ptrGtPSB,
                            nrL1NoIsoEG,
                            nrL1IsoEG,
                            nrL1CenJet,
                            nrL1ForJet,
                            nrL1TauJet,
                            ifCaloEtaNumberBits);

                    caloCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = caloCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        caloCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << myCout.str() << std::endl;
                    }
                    //                    delete caloCondition;

                }
                    break;
                case CondEnergySum: {
                    L1GtEnergySumCondition* eSumCondition = new L1GtEnergySumCondition(
                            itCond->second, ptrGtPSB);
                    eSumCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = eSumCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        eSumCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << myCout.str() << std::endl;
                    }
                    //                    delete eSumCondition;

                }
                    break;
                case CondJetCounts: {
                    L1GtJetCountsCondition* jcCondition = new L1GtJetCountsCondition(
                            itCond->second, ptrGtPSB, nrL1JetCounts);
                    jcCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = jcCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        jcCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << myCout.str() << std::endl;
                    }

                    //                  delete jcCondition;

                }
                    break;
                case CondCorrelation: {
                    //L1GtCorrelationCondition correlationCond = FIXME;

                }
                    break;
                case CondNull: {

                    // do nothing

                }
                    break;
                default: {

                    // do nothing

                }
                    break;
            }

        }

        conditionResultMaps.push_back(cMapResults);

    }

    // loop over algorithm map

    // empty vector for object maps - filled during loop
    std::vector<L1GlobalTriggerObjectMap> objMapVec;
    objMapVec.reserve(numberPhysTriggers);

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        L1GtAlgorithmEvaluation gtAlg(itAlgo->second);
        gtAlg.evaluateAlgorithm((itAlgo->second).algoChipNumber(), conditionResultMaps);

        int algBitNumber = (itAlgo->second).algoBitNumber();
        bool algResult = gtAlg.gtAlgoResult();

        if (algResult) {
            m_gtlAlgorithmOR.set(algBitNumber);
        }

        // object maps only for BxInEvent = 0
        if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {

            // set object map
            L1GlobalTriggerObjectMap objMap;

            objMap.setAlgoName(itAlgo->first);
            objMap.setAlgoBitNumber(algBitNumber);
            objMap.setAlgoGtlResult(algResult);
            objMap.setOperandTokenVector(gtAlg.operandTokenVector());
            objMap.setCombinationVector(*(gtAlg.gtAlgoCombinationVector()));

            if (edm::isDebugEnabled() ) {
                std::ostringstream myCout1;
                objMap.print(myCout1);

                LogTrace("L1GlobalTriggerGTL") << myCout1.str() << std::endl;
            }

            objMapVec.push_back(objMap);

        }

        if (edm::isDebugEnabled() ) {
            std::ostringstream myCout;
            (itAlgo->second).print(myCout);
            gtAlg.print(myCout);

            LogTrace("L1GlobalTriggerGTL") << myCout.str() << std::endl;
        }

    }

    // object maps only for BxInEvent = 0
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
        gtObjectMapRecord->setGtObjectMap(objMapVec);
    }

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // delete the conditions created with new, clear all
    for (std::vector<L1GtAlgorithmEvaluation::ConditionEvaluationMap>::iterator 
        itCondOnChip = conditionResultMaps.begin(); itCondOnChip != conditionResultMaps.end(); 
        itCondOnChip++) {

        for (L1GtAlgorithmEvaluation::ItEvalMap itCond =
            itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            if (itCond->second != 0) {
                delete itCond->second;
            }
            itCond->second = 0;

        }

        itCondOnChip->clear();
    }

    conditionResultMaps.clear();

}

// clear GTL
void L1GlobalTriggerGTL::reset() {

    m_candL1Mu->clear();

    m_gtlDecisionWord.reset();
    m_gtlAlgorithmOR.reset();

}

// print Global Muon Trigger data received by GTL
void L1GlobalTriggerGTL::printGmtData(const int iBxInEvent) const {

    LogTrace("L1GlobalTriggerGTL")
            << "\nL1GlobalTrigger: GMT data received for BxInEvent = "
            << iBxInEvent << std::endl;

    int nrL1Mu = m_candL1Mu->size();    
    LogTrace("L1GlobalTriggerGTL")
            << "Number of GMT muons = " << nrL1Mu << "\n" 
            << std::endl;

    for (std::vector<const L1MuGMTCand*>::const_iterator iter =
            m_candL1Mu->begin(); iter != m_candL1Mu->end(); iter++) {

        LogTrace("L1GlobalTriggerGTL") << *(*iter) << std::endl;

    }

    LogTrace("L1GlobalTriggerGTL") << std::endl;

}
