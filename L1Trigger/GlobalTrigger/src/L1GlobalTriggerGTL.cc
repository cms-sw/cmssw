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
#include <vector>
#include <ext/hash_map>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

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

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations

// constructor
L1GlobalTriggerGTL::L1GlobalTriggerGTL() :
    m_candL1Mu(new std::vector<L1MuGMTCand*>(L1GlobalTriggerReadoutSetup::NumberL1Muons)) {

    m_gtlAlgorithmOR.reset();
    m_gtlDecisionWord.reset();

    m_candL1Mu->reserve(L1GlobalTriggerReadoutSetup::NumberL1Muons);
    
    // initialize cached IDs
    m_l1GtMenuCacheID = 0ULL;
}

// destructor
L1GlobalTriggerGTL::~L1GlobalTriggerGTL() {

    reset();
    m_candL1Mu->clear();
    delete m_candL1Mu;

}

// operations

// receive data from Global Muon Trigger
void L1GlobalTriggerGTL::receiveGmtObjectData(edm::Event& iEvent,
    const edm::InputTag& muGmtInputTag, const int iBxInEvent, const bool receiveMu,
    const int nrL1Mu) {

    //
    reset();

    //
    if (receiveMu) {

        LogDebug("L1GlobalTriggerGTL")
            << "**** L1GlobalTriggerGTL receiving muon data from input tag "
            << muGmtInputTag << std::endl;

        // get data from Global Muon Trigger

        edm::Handle<std::vector<L1MuGMTCand> > muonData;
        iEvent.getByLabel(muGmtInputTag, muonData);

        for (int iMuon = 0; iMuon < nrL1Mu; iMuon++) {

            L1MuGMTCand muCand;
            int nMuon = 0;

            std::vector< L1MuGMTCand>::const_iterator itMuon;
            for (itMuon = muonData->begin(); itMuon != muonData->end(); itMuon++) {

                // retrieving info for a given bx only
                if ((*itMuon).bx() == iBxInEvent) {
                    if (nMuon == iMuon) {
                        muCand = (*itMuon);
                        break;
                    }
                    nMuon++;
                }
            }

            (*m_candL1Mu)[iMuon] = new L1MuGMTCand( muCand );
        }

    }
    else {

        LogDebug("L1GlobalTriggerGTL") << "\n**** Global Muon input disabled!"
            << "     All candidates empty." << "\n**** \n" << std::endl;

        // set all muon candidates empty
        for (int iMuon = 0; iMuon < nrL1Mu; iMuon++) {

            MuonDataWord dataword = 0;
            (*m_candL1Mu)[iMuon] = new L1MuGMTCand( dataword );
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

    std::vector<L1MuGMTCand*>::iterator iter;
    for (iter = m_candL1Mu->begin(); iter < m_candL1Mu->end(); iter++) {
        if (*iter) {
            delete (*iter);
            *iter = 0;
        }
    }

    m_gtlDecisionWord.reset();

    m_gtlAlgorithmOR.reset();

}

// print Global Muon Trigger data received by GTL
void L1GlobalTriggerGTL::printGmtData(int iBxInEvent) const {

    LogTrace("L1GlobalTriggerGTL") << "\nMuon data received by GTL for BxInEvent = " << iBxInEvent
        << std::endl;

    for (std::vector<L1MuGMTCand*>::iterator iter = m_candL1Mu->begin(); iter < m_candL1Mu->end(); iter++) {

        LogTrace("L1GlobalTriggerGTL") << std::endl;

        (*iter)->print();

    }

    LogTrace("L1GlobalTriggerGTL") << std::endl;

}
