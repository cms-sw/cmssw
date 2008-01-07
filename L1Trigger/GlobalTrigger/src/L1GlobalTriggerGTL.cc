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

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtMuonCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCaloCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtEnergySumCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtJetCountsCondition.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GtCorrelationCondition.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerMuonTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerCaloTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerEsumsTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerJetCountsTemplate.h"

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
L1GlobalTriggerGTL::L1GlobalTriggerGTL(const L1GlobalTrigger& gt) :
    m_GT(gt), m_candL1Mu(new L1GmtCandVector(L1GlobalTriggerReadoutSetup::NumberL1Muons)) {

    m_gtlAlgorithmOR.reset();
    m_gtlDecisionWord.reset();

    m_candL1Mu->reserve(L1GlobalTriggerReadoutSetup::NumberL1Muons);

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
    const unsigned int nrL1Mu) {

    //
    reset();

    //
    if (receiveMu) {

        LogDebug("L1GlobalTriggerGTL")
            << "**** L1GlobalTriggerGTL receiving muon data from input tag "
            << muGmtInputTag.label() << std::endl;

        // get data from Global Muon Trigger

        edm::Handle<std::vector<L1MuGMTCand> > muonData;
        iEvent.getByLabel(muGmtInputTag.label(), muonData);

        for (unsigned int iMuon = 0; iMuon < nrL1Mu; iMuon++) {

            L1MuGMTCand muCand;
            unsigned int nMuon = 0;

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
        for (unsigned int iMuon = 0; iMuon < nrL1Mu; iMuon++) {

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
    const unsigned int numberPhysTriggers) {

    //    LogDebug ("Trace") << "**** L1GlobalTriggerGTL run " << std::endl;

    // try xml conditions
    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    if (gtConf != 0) {
        unsigned int chipnr;
        LogDebug("L1GlobalTriggerGTL") << "\n***** Result of the XML-conditions \n" << std::endl;

        for (chipnr = 0; chipnr < L1GlobalTriggerConfig::NumberConditionChips; chipnr++) {
            LogTrace("L1GlobalTriggerGTL") << "\n---------Chip " << chipnr + 1 << " ----------\n"
                << std::endl;

            for (L1GlobalTriggerConfig::ConditionsMap::const_iterator itxml =
                gtConf->conditionsmap[chipnr].begin(); itxml != gtConf->conditionsmap[chipnr].end(); itxml++) {

                std::string condName = itxml->first;

                LogTrace("L1GlobalTriggerGTL")
                    << "\n===============================================\n"
                    << "Evaluating condition: " << condName << "\n" << std::endl;

                bool condResult = itxml->second->blockCondition_sr();

                LogTrace("L1GlobalTriggerGTL") << condName << " result: " << condResult
                    << std::endl;

            }
        }

        LogTrace("L1GlobalTriggerGTL") << "\n---------- Prealgorithms: evaluation ---------\n"
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator itxml =
            gtConf->prealgosmap.begin(); itxml != gtConf->prealgosmap.end(); itxml++) {

            std::string prealgoName = itxml->first;
            bool prealgoResult = itxml->second->blockCondition_sr();
            std::string prealgoLogExpression = itxml->second->getLogicalExpression();
            std::string prealgoNumExpression = itxml->second->getNumericExpression();

            // set the pins if VERSION_FINAL (final version uses prealgorithms)
            if (gtConf->getVersion() == L1GlobalTriggerConfig::VERSION_FINAL) {

                // algo( i ) = prealgo( i+1 ), i = 0, MaxNumberAlgorithms
                int prealgoNumber = itxml->second->getAlgoNumber();

                if (itxml->second->getLastResult()) {
                    if (prealgoNumber > 0) {
                        m_gtlAlgorithmOR.set(prealgoNumber-1);

                    }
                }

                LogTrace("L1GlobalTriggerGTL") << " Bit " << prealgoNumber-1 << " " << prealgoName
                    << " = " << prealgoLogExpression << ": " << m_gtlAlgorithmOR[ prealgoNumber-1 ]
                    << " = " << prealgoNumExpression << std::endl;
            }
            else {

                LogTrace("L1GlobalTriggerGTL") << " " << prealgoName << " = "
                    << prealgoLogExpression << ": " << prealgoResult << " = "
                    << prealgoNumExpression << std::endl;
            }

        }
        LogTrace("L1GlobalTriggerGTL") << "\n---------- End of prealgorithm list ---------\n"
            << std::endl;

        LogTrace("L1GlobalTriggerGTL") << "\n---------- Algorithms: evaluation ----------\n"
            << std::endl;
        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator itxml = gtConf->algosmap.begin(); itxml
            != gtConf->algosmap.end(); itxml++) {

            std::string algoName = itxml->first;
            bool algoResult = itxml->second->blockCondition_sr();
            std::string algoLogExpression = itxml->second->getLogicalExpression();
            std::string algoNumExpression = itxml->second->getNumericExpression();

            // set the pins if VERSION_PROTOTYPE (prototype version uses algos)
            if (gtConf->getVersion() == L1GlobalTriggerConfig::VERSION_PROTOTYPE) {
                if (itxml->second->getLastResult()) {
                    if (itxml->second->getOutputPin() > 0) {
                        m_gtlAlgorithmOR.set(itxml->second->getOutputPin()-1);
                    }
                }

                LogTrace("L1GlobalTriggerGTL") << " Bit " << itxml->second->getOutputPin()-1 << " "
                    << algoName << " = " << algoLogExpression << " = " << algoNumExpression
                    << std::endl;

            }
            else {

                LogTrace("L1GlobalTriggerGTL") << "  " << algoName << " = " << algoLogExpression
                    << " = " << algoNumExpression << " = " << algoResult << std::endl;
            }

        }
        LogTrace("L1GlobalTriggerGTL") << "\n---------- End of algorithm list ----------\n"
            << std::endl;

    }

    // FIXME remove it after comparison
    std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> TMP_gtlAlgorithmOR =
        m_gtlAlgorithmOR;
    m_gtlAlgorithmOR.reset();

    // get the trigger menu from the EventSetup

    edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
    evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;

    std::vector<ConditionMap> conditionMap = l1GtMenu->gtConditionMap();
    AlgorithmMap algorithmMap = l1GtMenu->gtAlgorithmMap();

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // save the results in temporary maps

    std::vector<std::map<std::string, L1GtConditionEvaluation*> >
        conditionResultMaps(conditionMap.size());

    int iMap = 0;

    for (std::vector<ConditionMap>::iterator itCondOnChip = conditionMap.begin(); itCondOnChip
        != conditionMap.end(); itCondOnChip++) {

        for (ItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            // evaluate condition
            switch ((itCond->second)->condCategory()) {
                case CondMuon: {

                    L1GtMuonCondition* muCondition = new L1GtMuonCondition(itCond->second, this, evSetup);
                    muCondition->evaluateConditionStoreResult();

                    (conditionResultMaps.at(iMap))[itCond->first] = muCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        muCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << "NEW " << myCout.str() << std::endl;
                    }

                    //delete muCondition;

                }
                    break;
                case CondCalo: {

                    L1GtCaloCondition* caloCondition = new L1GtCaloCondition(itCond->second, ptrGtPSB, evSetup);
                    caloCondition->evaluateConditionStoreResult();

                    (conditionResultMaps.at(iMap))[itCond->first] = caloCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        caloCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << "NEW " << myCout.str() << std::endl;
                    }
                    //                    delete caloCondition;

                }
                    break;
                case CondEnergySum: {
                    L1GtEnergySumCondition* eSumCondition = new L1GtEnergySumCondition(itCond->second, ptrGtPSB, evSetup);
                    eSumCondition->evaluateConditionStoreResult();

                    (conditionResultMaps.at(iMap))[itCond->first] = eSumCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        eSumCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << "NEW " << myCout.str() << std::endl;
                    }
                    //                    delete eSumCondition;

                }
                    break;
                case CondJetCounts: {
                    L1GtJetCountsCondition* jcCondition = new L1GtJetCountsCondition(itCond->second, ptrGtPSB, evSetup);
                    jcCondition->evaluateConditionStoreResult();

                    (conditionResultMaps.at(iMap))[itCond->first] = jcCondition;

                    if (edm::isDebugEnabled() ) {
                        std::ostringstream myCout;
                        jcCondition->print(myCout);

                        LogTrace("L1GlobalTriggerGTL") << "NEW " << myCout.str() << std::endl;
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

        iMap++;

    }

    // loop over algorithm map

    // empty vector for object maps - filled during loop
    std::vector<L1GlobalTriggerObjectMap> objMapVec;
    objMapVec.reserve(numberPhysTriggers);

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        L1GtAlgorithmEvaluation gtAlg(itAlgo->second);
        gtAlg.evaluateAlgorithm((itAlgo->second)->algoChipNumber(), conditionResultMaps);

        int algBitNumber = (itAlgo->second)->algoBitNumber();
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
            objMap.setAlgoLogicalExpression(gtAlg.logicalExpression());
            objMap.setAlgoNumericalExpression(gtAlg.gtAlgoNumericalExpression());
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
            (itAlgo->second)->print(myCout);
            gtAlg.print(myCout);

            LogTrace("L1GlobalTriggerGTL") << "NEW " << myCout.str() << std::endl;
        }

    }

    // object maps only for BxInEvent = 0
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
        gtObjectMapRecord->setGtObjectMap(objMapVec);
    }

    if (edm::isDebugEnabled() ) {
        // FIXME remove it after comparison

        LogTrace("L1GlobalTriggerGTL") << TMP_gtlAlgorithmOR << std::endl;
        LogTrace("L1GlobalTriggerGTL") << m_gtlAlgorithmOR << std::endl;

        if (TMP_gtlAlgorithmOR == m_gtlAlgorithmOR) {
            LogTrace("L1GlobalTriggerGTL") << "COMPARISON OLD - NEW OK " << std::endl;

        }
        else {
            LogTrace("L1GlobalTriggerGTL") << "COMPARISON OLD - NEW: NOT MATCH " << std::endl;

        }
    }

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // delete the conditions created with new, clear all
    for (std::vector<std::map<std::string, L1GtConditionEvaluation*> >::iterator itCondOnChip =
        conditionResultMaps.begin(); itCondOnChip != conditionResultMaps.end(); itCondOnChip++) {

        for (std::map<std::string, L1GtConditionEvaluation*>::iterator itCond =
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



// fill object map record
const std::vector<L1GlobalTriggerObjectMap>* L1GlobalTriggerGTL::objectMap() {

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();

    // empty vector for object maps
    std::vector<L1GlobalTriggerObjectMap>* objMapVec = new std::vector<L1GlobalTriggerObjectMap>();

    // do it only if VERSION_FINAL
    if (gtConf->getVersion() == L1GlobalTriggerConfig::VERSION_FINAL) {

        L1GlobalTriggerObjectMap* objMap = new L1GlobalTriggerObjectMap();

        for (L1GlobalTriggerConfig::ConditionsMap::const_iterator itxml =
            gtConf->prealgosmap.begin(); itxml != gtConf->prealgosmap.end(); itxml++) {

            std::string prealgoNameVal = itxml->first;

            // algo( i ) = prealgo( i+1 ), i = 0, MaxNumberAlgorithms
            int prealgoNumberVal = itxml->second->getAlgoNumber();
            int algoBitNumberVal = prealgoNumberVal -1;

            bool prealgoResultVal = itxml->second->getLastResult();

            std::string prealgoLogExpressionVal = itxml->second->getLogicalExpression();
            std::string prealgoNumExpressionVal = itxml->second->getNumericExpression();

            std::vector<CombinationsInCond> combVectorVal = itxml->second->getCombinationVector();

            std::vector<ObjectTypeInCond> objTypeVal = itxml->second->getObjectTypeVector();

            // set object map

            objMap->reset();

            objMap->setAlgoName(prealgoNameVal);
            objMap->setAlgoBitNumber(algoBitNumberVal);
            objMap->setAlgoGtlResult(prealgoResultVal);
            objMap->setAlgoLogicalExpression(prealgoLogExpressionVal);
            objMap->setAlgoNumericalExpression(prealgoNumExpressionVal);
            objMap->setCombinationVector(combVectorVal);
            objMap->setObjectTypeVector(objTypeVal);

            if (edm::isDebugEnabled() ) {
                std::ostringstream myCout1;
                objMap->print(myCout1);

                LogTrace("L1GlobalTriggerGTL") << myCout1.str() << std::endl;
            }

            objMapVec->push_back(*objMap);

            // after last usage, objMapVec must be deleted!

        }

        delete objMap;
    }

    return objMapVec;

}

// clear GTL
void L1GlobalTriggerGTL::reset() {

    L1GmtCandVector::iterator iter;
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

    for (L1GmtCandVector::iterator iter = m_candL1Mu->begin(); iter < m_candL1Mu->end(); iter++) {

        LogTrace("L1GlobalTriggerGTL") << std::endl;

        (*iter)->print();

    }

    LogTrace("L1GlobalTriggerGTL") << std::endl;

}
