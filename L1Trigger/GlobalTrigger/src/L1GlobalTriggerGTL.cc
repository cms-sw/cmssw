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
#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtConditionEvaluation.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtMuonCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCaloCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtEnergySumCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtJetCountsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtHfBitCountsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtHfRingEtSumsCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCastorCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtBptxCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtExternalCondition.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtCorrelationCondition.h"

#include "L1Trigger/GlobalTrigger/interface/L1GtEtaPhiConversions.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations

// constructor
L1GlobalTriggerGTL::L1GlobalTriggerGTL(const edm::InputTag & m_muGmtInputTag,edm::ConsumesCollector && iC) :
    m_candL1Mu( new std::vector<const L1MuGMTCand*>),
    m_isDebugEnabled(edm::isDebugEnabled())
{

    m_gtlAlgorithmOR.reset();
    m_gtlDecisionWord.reset();

    // initialize cached IDs
    m_l1GtMenuCacheID = 0ULL;
    m_l1CaloGeometryCacheID = 0ULL;
    m_l1MuTriggerScalesCacheID = 0ULL;

    // pointer to conversion - actually done in the event loop (cached)
    m_gtEtaPhiConversions = new L1GtEtaPhiConversions();
    m_gtEtaPhiConversions->setVerbosity(m_verbosity);

    iC.consumes<std::vector<L1MuGMTCand> >(m_muGmtInputTag);

}

// destructor
L1GlobalTriggerGTL::~L1GlobalTriggerGTL() {

    reset();
    delete m_candL1Mu;
    delete m_gtEtaPhiConversions;

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

    if (m_verbosity) {
        LogDebug("L1GlobalTrigger")
                << "\n**** L1GlobalTriggerGTL receiving muon data for BxInEvent = "
                << iBxInEvent << "\n     from input tag " << muGmtInputTag << "\n"
                << std::endl;

    }

    reset();

    // get data from Global Muon Trigger
    if (receiveMu) {

        edm::Handle<std::vector<L1MuGMTCand> > muonData;
        iEvent.getByLabel(muGmtInputTag, muonData);

        if (!muonData.isValid()) {
            if (m_verbosity) {
                edm::LogWarning("L1GlobalTrigger")
                        << "\nWarning: std::vector<L1MuGMTCand> with input tag "
                        << muGmtInputTag
                        << "\nrequested in configuration, but not found in the event.\n"
                        << std::endl;
            }
        } else {

            std::vector<L1MuGMTCand>::const_iterator itMuon;
            for (itMuon = muonData->begin(); itMuon != muonData->end(); itMuon++) {
                if ((*itMuon).bx() == iBxInEvent) {

                    (*m_candL1Mu).push_back(&(*itMuon));
                    //LogTrace("L1GlobalTrigger") << (*itMuon)
                    //        << std::endl;

                }

            }

        }
    }

    if (m_verbosity && m_isDebugEnabled) {
        printGmtData(iBxInEvent);
    }

}

// run GTL
void L1GlobalTriggerGTL::run(
        edm::Event& iEvent, const edm::EventSetup& evSetup,
        const L1GlobalTriggerPSB* ptrGtPSB,
        const bool produceL1GtObjectMapRecord,
        const int iBxInEvent,
        std::auto_ptr<L1GlobalTriggerObjectMapRecord>& gtObjectMapRecord,
        const unsigned int numberPhysTriggers,
        const int nrL1Mu,
        const int nrL1NoIsoEG, const int nrL1IsoEG,
        const int nrL1CenJet, const int nrL1ForJet, const int nrL1TauJet,
        const int nrL1JetCounts,
        const int ifMuEtaNumberBits, const int ifCaloEtaNumberBits) {


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

    const std::vector<std::vector<L1GtMuonTemplate> >& corrMuon =
            m_l1GtMenu->corMuonTemplate();

    const std::vector<std::vector<L1GtCaloTemplate> >& corrCalo =
            m_l1GtMenu->corCaloTemplate();

    const std::vector<std::vector<L1GtEnergySumTemplate> >& corrEnergySum =
            m_l1GtMenu->corEnergySumTemplate();

    // conversion needed for correlation conditions
    // done in the condition loop when the first correlation template is in the menu
    bool convertScale = false;

    // get / update the calorimeter geometry from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1CaloGeometryCacheID =
            evSetup.get<L1CaloGeometryRecord>().cacheIdentifier();

    if (m_l1CaloGeometryCacheID != l1CaloGeometryCacheID) {

        edm::ESHandle<L1CaloGeometry> l1CaloGeometry;
        evSetup.get<L1CaloGeometryRecord>().get(l1CaloGeometry) ;
        m_l1CaloGeometry =  l1CaloGeometry.product();

        m_l1CaloGeometryCacheID = l1CaloGeometryCacheID;
        convertScale = true;

    }

    // get / update the eta and phi muon trigger scales from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1MuTriggerScalesCacheID =
            evSetup.get<L1MuTriggerScalesRcd>().cacheIdentifier();

    if (m_l1MuTriggerScalesCacheID != l1MuTriggerScalesCacheID) {

        edm::ESHandle< L1MuTriggerScales> l1MuTriggerScales;
        evSetup.get< L1MuTriggerScalesRcd>().get(l1MuTriggerScales);
        m_l1MuTriggerScales = l1MuTriggerScales.product();

        m_l1MuTriggerScalesCacheID = l1MuTriggerScalesCacheID;
        convertScale = true;
    }

    if (convertScale) {
        m_gtEtaPhiConversions->setVerbosity(m_verbosity);
        m_gtEtaPhiConversions->convertL1Scales(m_l1CaloGeometry,
                m_l1MuTriggerScales, ifCaloEtaNumberBits, ifMuEtaNumberBits);

        // print the conversions if DEBUG and verbosity enabled

        if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            m_gtEtaPhiConversions->print(myCout);

            LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
        }

        // set convertScale to false to avoid executing the conversion
        // more than once - in case the scales change it will be set to true
        // in the cache check
        convertScale = false;
    }


    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // save the results in temporary maps

    // never happens in production but at first event...
    if (m_conditionResultMaps.size() != conditionMap.size()) {
        m_conditionResultMaps.clear();
        m_conditionResultMaps.resize(conditionMap.size());
    }
    
    int iChip = -1;

    for (std::vector<ConditionMap>::const_iterator
    		itCondOnChip = conditionMap.begin(); itCondOnChip != conditionMap.end(); itCondOnChip++) {

        iChip++;

        //L1GtAlgorithmEvaluation::ConditionEvaluationMap cMapResults;
        // L1GtAlgorithmEvaluation::ConditionEvaluationMap cMapResults((*itCondOnChip).size()); // hash map
        L1GtAlgorithmEvaluation::ConditionEvaluationMap& cMapResults =
                m_conditionResultMaps[iChip];

        for (CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {

            // evaluate condition
            switch ((itCond->second)->condCategory()) {
                case CondMuon: {

                    L1GtMuonCondition* muCondition = new L1GtMuonCondition(itCond->second, this,
                            nrL1Mu, ifMuEtaNumberBits);

                    muCondition->setVerbosity(m_verbosity);
                    muCondition->setGtCorrParDeltaPhiNrBins(
                            (m_gtEtaPhiConversions->gtObjectNrBinsPhi(Mu)) / 2
                                    + 1);
                    muCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = muCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        muCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
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

                    caloCondition->setVerbosity(m_verbosity);
                    caloCondition->setGtCorrParDeltaPhiNrBins(
                            (m_gtEtaPhiConversions->gtObjectNrBinsPhi(
                                    ((itCond->second)->objectType())[0])) / 2
                                    + 1);
                    caloCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = caloCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        caloCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }
                    //                    delete caloCondition;

                }
                    break;
                case CondEnergySum: {
                    L1GtEnergySumCondition* eSumCondition = new L1GtEnergySumCondition(
                            itCond->second, ptrGtPSB);

                    eSumCondition->setVerbosity(m_verbosity);
                    eSumCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = eSumCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        eSumCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }
                    //                    delete eSumCondition;

                }
                    break;
                case CondJetCounts: {
                    L1GtJetCountsCondition* jcCondition = new L1GtJetCountsCondition(
                            itCond->second, ptrGtPSB, nrL1JetCounts);

                    jcCondition->setVerbosity(m_verbosity);
                    jcCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = jcCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        jcCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete jcCondition;

                }
                    break;
                case CondHfBitCounts: {
                    L1GtHfBitCountsCondition* bcCondition = new L1GtHfBitCountsCondition(
                            itCond->second, ptrGtPSB);

                    bcCondition->setVerbosity(m_verbosity);
                    bcCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = bcCondition;

                    if (m_isDebugEnabled ) {
                        std::ostringstream myCout;
                        bcCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete bcCondition;

                }
                    break;
                case CondHfRingEtSums: {
                    L1GtHfRingEtSumsCondition* etCondition = new L1GtHfRingEtSumsCondition(
                            itCond->second, ptrGtPSB);

                    etCondition->setVerbosity(m_verbosity);
                    etCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = etCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        etCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete etCondition;

                }
                    break;
                case CondCastor: {
                    bool castorCondResult = false;

                    // FIXME need a solution to read CASTOR
                    //if (castorConditionFlag) {
                    //    castorCondResult = castorData->conditionResult(itCond->first);
                    //}

                    L1GtCastorCondition* castorCondition = new L1GtCastorCondition(
                            itCond->second, castorCondResult);

                    castorCondition->setVerbosity(m_verbosity);
                    castorCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = castorCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        castorCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete castorCondition;

                }
                    break;
                case CondBptx: {
                    bool bptxCondResult = true;

                    // FIXME need a solution to read BPTX with real value

                    L1GtBptxCondition* bptxCondition = new L1GtBptxCondition(
                            itCond->second, bptxCondResult);

                    bptxCondition->setVerbosity(m_verbosity);
                    bptxCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = bptxCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        bptxCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete bptxCondition;

                }
                    break;
                case CondExternal: {
                    bool externalCondResult = true;

                    // FIXME need a solution to read External with real value

                    L1GtExternalCondition* externalCondition = new L1GtExternalCondition(
                            itCond->second, externalCondResult);

                    externalCondition->setVerbosity(m_verbosity);
                    externalCondition->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = externalCondition;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        externalCondition->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete externalCondition;

                }
                    break;
                case CondCorrelation: {

                    // get first the sub-conditions
                    const L1GtCorrelationTemplate* corrTemplate =
                        static_cast<const L1GtCorrelationTemplate*>(itCond->second);
                    const L1GtConditionCategory cond0Categ = corrTemplate->cond0Category();
                    const L1GtConditionCategory cond1Categ = corrTemplate->cond1Category();
                    const int cond0Ind = corrTemplate->cond0Index();
                    const int cond1Ind = corrTemplate->cond1Index();

                    const L1GtCondition* cond0Condition = 0;
                    const L1GtCondition* cond1Condition = 0;

                    // maximum number of objects received for evaluation of Type1s condition
                    int cond0NrL1Objects = 0;
                    int cond1NrL1Objects = 0;

                    int cond0EtaBits = 0;
                    int cond1EtaBits = 0;

                    switch (cond0Categ) {
                        case CondMuon: {
                            cond0Condition = &((corrMuon[iChip])[cond0Ind]);
                            cond0NrL1Objects = nrL1Mu;
                            cond0EtaBits = ifMuEtaNumberBits;
                        }
                            break;
                        case CondCalo: {
                            cond0Condition = &((corrCalo[iChip])[cond0Ind]);

                            switch ((cond0Condition->objectType())[0]) {
                                case NoIsoEG:
                                    cond0NrL1Objects= nrL1NoIsoEG;
                                    break;
                                case IsoEG:
                                    cond0NrL1Objects = nrL1IsoEG;
                                    break;
                                case CenJet:
                                    cond0NrL1Objects = nrL1CenJet;
                                    break;
                                case ForJet:
                                    cond0NrL1Objects = nrL1ForJet;
                                    break;
                                case TauJet:
                                    cond0NrL1Objects = nrL1TauJet;
                                    break;
                                default:
                                    cond0NrL1Objects = 0;
                                    break;
                            }

                            cond0EtaBits = ifCaloEtaNumberBits;
                        }
                            break;
                        case CondEnergySum: {
                            cond0Condition = &((corrEnergySum[iChip])[cond0Ind]);
                            cond0NrL1Objects = 1;
                        }
                            break;
                        default: {
                            // do nothing, should not arrive here
                        }
                            break;
                    }

                    switch (cond1Categ) {
                        case CondMuon: {
                            cond1Condition = &((corrMuon[iChip])[cond1Ind]);
                            cond1NrL1Objects = nrL1Mu;
                            cond1EtaBits = ifMuEtaNumberBits;
                        }
                            break;
                        case CondCalo: {
                            cond1Condition = &((corrCalo[iChip])[cond1Ind]);

                            switch ((cond1Condition->objectType())[0]) {
                                case NoIsoEG:
                                    cond1NrL1Objects= nrL1NoIsoEG;
                                    break;
                                case IsoEG:
                                    cond1NrL1Objects = nrL1IsoEG;
                                    break;
                                case CenJet:
                                    cond1NrL1Objects = nrL1CenJet;
                                    break;
                                case ForJet:
                                    cond1NrL1Objects = nrL1ForJet;
                                    break;
                                case TauJet:
                                    cond1NrL1Objects = nrL1TauJet;
                                    break;
                                default:
                                    cond1NrL1Objects = 0;
                                    break;
                            }

                             cond1EtaBits = ifCaloEtaNumberBits;
                        }
                            break;
                        case CondEnergySum: {
                            cond1Condition = &((corrEnergySum[iChip])[cond1Ind]);
                            cond1NrL1Objects = 1;
                        }
                            break;
                        default: {
                            // do nothing, should not arrive here
                        }
                            break;
                    }

                    L1GtCorrelationCondition* correlationCond =
                        new L1GtCorrelationCondition(itCond->second,
                            cond0Condition, cond1Condition,
                            cond0NrL1Objects, cond1NrL1Objects,
                            cond0EtaBits, cond1EtaBits,
                            this, ptrGtPSB, m_gtEtaPhiConversions);

                    correlationCond->setVerbosity(m_verbosity);
                    correlationCond->evaluateConditionStoreResult();

                    cMapResults[itCond->first] = correlationCond;

                    if (m_verbosity && m_isDebugEnabled) {
                        std::ostringstream myCout;
                        correlationCond->print(myCout);

                        LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
                    }

                    //                  delete correlationCond;

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

    }

    // loop over algorithm map

    // empty vector for object maps - filled during loop
    std::vector<L1GlobalTriggerObjectMap> objMapVec;
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) objMapVec.reserve(numberPhysTriggers);

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        L1GtAlgorithmEvaluation gtAlg(itAlgo->second);
        gtAlg.evaluateAlgorithm((itAlgo->second).algoChipNumber(), m_conditionResultMaps);

        int algBitNumber = (itAlgo->second).algoBitNumber();
        bool algResult = gtAlg.gtAlgoResult();

        if (algResult) {
            m_gtlAlgorithmOR.set(algBitNumber);
        }

        if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCout;
            ( itAlgo->second ).print(myCout);
            gtAlg.print(myCout);

            LogTrace("L1GlobalTrigger") << myCout.str() << std::endl;
        }


        // object maps only for BxInEvent = 0
        if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {

            // set object map
            L1GlobalTriggerObjectMap objMap;

            objMap.setAlgoName(itAlgo->first);
            objMap.setAlgoBitNumber(algBitNumber);
            objMap.setAlgoGtlResult(algResult);
            objMap.swapOperandTokenVector(gtAlg.operandTokenVector());
            objMap.swapCombinationVector(gtAlg.gtAlgoCombinationVector());
	    // gtAlg is empty now...

            if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCout1;
                objMap.print(myCout1);

                LogTrace("L1GlobalTrigger") << myCout1.str() << std::endl;
            }

            objMapVec.push_back(objMap);

        }


    }

    // object maps only for BxInEvent = 0
    if (produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
        gtObjectMapRecord->swapGtObjectMap(objMapVec);
    }

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    // delete the conditions created with new, zero pointer, do not clear map, keep the vector as is...
    for (std::vector<L1GtAlgorithmEvaluation::ConditionEvaluationMap>::iterator
            itCondOnChip  = m_conditionResultMaps.begin();
            itCondOnChip != m_conditionResultMaps.end(); itCondOnChip++) {

        for (L1GtAlgorithmEvaluation::ItEvalMap
                itCond  = itCondOnChip->begin();
                itCond != itCondOnChip->end(); itCond++) {

            delete itCond->second;
            itCond->second = 0;
        }
    }

}

// clear GTL
void L1GlobalTriggerGTL::reset() {

    m_candL1Mu->clear();

    m_gtlDecisionWord.reset();
    m_gtlAlgorithmOR.reset();

}

// print Global Muon Trigger data received by GTL
void L1GlobalTriggerGTL::printGmtData(const int iBxInEvent) const {

    LogTrace("L1GlobalTrigger")
            << "\nL1GlobalTrigger: GMT data received for BxInEvent = "
            << iBxInEvent << std::endl;

    int nrL1Mu = m_candL1Mu->size();
    LogTrace("L1GlobalTrigger")
            << "Number of GMT muons = " << nrL1Mu << "\n"
            << std::endl;

    for (std::vector<const L1MuGMTCand*>::const_iterator iter =
            m_candL1Mu->begin(); iter != m_candL1Mu->end(); iter++) {

        LogTrace("L1GlobalTrigger") << *(*iter) << std::endl;

    }

    LogTrace("L1GlobalTrigger") << std::endl;

}
