/**
 * \class L1GtUtils
 * 
 * 
 * Description: various methods for L1 GT, to be called in an EDM analyzer, producer or filter.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// system include files
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor(s)
L1GtUtils::L1GtUtils() :

    m_l1GtStableParCacheID(0ULL), m_numberPhysTriggers(0),

    m_numberTechnicalTriggers(0),

    m_l1GtPfAlgoCacheID(0ULL), m_l1GtPfTechCacheID(0ULL),

    m_l1GtTmAlgoCacheID(0ULL), m_l1GtTmTechCacheID(0ULL),

    m_l1GtTmVetoAlgoCacheID(0ULL), m_l1GtTmVetoTechCacheID(0ULL),

    m_l1GtMenuCacheID(0ULL),

    m_physicsDaqPartition(0) {

    // empty
}

// destructor
L1GtUtils::~L1GtUtils() {

    // empty

}

void L1GtUtils::retrieveL1EventSetup(const edm::EventSetup& evSetup) {

    // get / update the stable parameters from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID =
            evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {

        edm::ESHandle<L1GtStableParameters> l1GtStablePar;
        evSetup.get<L1GtStableParametersRcd>().get(l1GtStablePar);
        m_l1GtStablePar = l1GtStablePar.product();

        // number of physics triggers
        m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of technical triggers
        m_numberTechnicalTriggers =
                m_l1GtStablePar->gtNumberTechnicalTriggers();

        int maxNumberTrigger = std::max(m_numberPhysTriggers,
                m_numberTechnicalTriggers);

        m_triggerMaskSet.reserve(maxNumberTrigger);
        m_prescaleFactorSet.reserve(maxNumberTrigger);

        //
        m_l1GtStableParCacheID = l1GtStableParCacheID;

    }

    // get / update the prescale factors from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtPfAlgoCacheID =
            evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);
        m_l1GtPfAlgo = l1GtPfAlgo.product();

        m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());

        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }

    unsigned long long l1GtPfTechCacheID = evSetup.get<
            L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

    if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfTech;
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().get(l1GtPfTech);
        m_l1GtPfTech = l1GtPfTech.product();

        m_prescaleFactorsTechTrig = &(m_l1GtPfTech->gtPrescaleFactors());

        m_l1GtPfTechCacheID = l1GtPfTechCacheID;

    }

    // get / update the trigger mask from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtTmAlgoCacheID =
            evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmAlgo;
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().get(l1GtTmAlgo);
        m_l1GtTmAlgo = l1GtTmAlgo.product();

        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();

        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }

    unsigned long long l1GtTmTechCacheID =
            evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmTech;
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().get(l1GtTmTech);
        m_l1GtTmTech = l1GtTmTech.product();

        m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();

        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }

    unsigned long long l1GtTmVetoAlgoCacheID =
            evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoAlgo;
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().get(l1GtTmVetoAlgo);
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

        m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();

        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }

    unsigned long long l1GtTmVetoTechCacheID =
            evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoTechCacheID != l1GtTmVetoTechCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoTech;
        evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().get(l1GtTmVetoTech);
        m_l1GtTmVetoTech = l1GtTmVetoTech.product();

        m_triggerMaskVetoTechTrig = m_l1GtTmVetoTech->gtTriggerMask();

        m_l1GtTmVetoTechCacheID = l1GtTmVetoTechCacheID;

    }

    // get / update the trigger menu from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtMenuCacheID =
            evSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {

        edm::ESHandle<L1GtTriggerMenu> l1GtMenu;
        evSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu);
        m_l1GtMenu = l1GtMenu.product();

        m_algorithmMap = &(m_l1GtMenu->gtAlgorithmMap());
        m_algorithmAliasMap = &(m_l1GtMenu->gtAlgorithmAliasMap());

        m_technicalTriggerMap = &(m_l1GtMenu->gtTechnicalTriggerMap());

        m_l1GtMenuCacheID = l1GtMenuCacheID;

    }

}

void L1GtUtils::getInputTag(const edm::Event& iEvent,
        edm::InputTag& l1GtRecordInputTag,
        edm::InputTag& l1GtReadoutRecordInputTag) {

    typedef std::vector<edm::Provenance const*> Provenances;
    Provenances provenances;
    std::string friendlyName;
    std::string modLabel;
    std::string instanceName;
    std::string processName;

    // to be sure that the input tags are correctly initialized
    edm::InputTag l1GtRecordInputTagVal;
    edm::InputTag l1GtReadoutRecordInputTagVal;

    bool foundL1GtRecord = false;
    bool foundL1GtReadoutRecord = false;

    iEvent.getAllProvenance(provenances);

    //edm::LogVerbatim("L1GtUtils") << "\n" << "Event contains "
    //        << provenances.size() << " product" << (provenances.size()==1 ? "" : "s")
    //        << " with friendlyClassName, moduleLabel, productInstanceName and processName:"
    //        << std::endl;

    for (Provenances::iterator itProv = provenances.begin(), itProvEnd =
            provenances.end(); itProv != itProvEnd; ++itProv) {

        friendlyName = (*itProv)->friendlyClassName();
        modLabel = (*itProv)->moduleLabel();
        instanceName = (*itProv)->productInstanceName();
        processName = (*itProv)->processName();

        //edm::LogVerbatim("L1GtUtils") << friendlyName << " \"" << modLabel
        //        << "\" \"" << instanceName << "\" \"" << processName << "\""
        //        << std::endl;

        if (friendlyName == "L1GlobalTriggerRecord") {
            l1GtRecordInputTagVal = edm::InputTag(modLabel, instanceName,
                    processName);
            foundL1GtRecord = true;
        } else if (friendlyName == "L1GlobalTriggerReadoutRecord") {

            l1GtReadoutRecordInputTagVal = edm::InputTag(modLabel, instanceName,
                    processName);
            foundL1GtReadoutRecord = true;
        }
    }

    // copy the input tags found to the returned arguments
    l1GtRecordInputTag = l1GtRecordInputTagVal;
    l1GtReadoutRecordInputTag = l1GtReadoutRecordInputTagVal;

    if (foundL1GtRecord) {
        edm::LogVerbatim("L1GtUtils")
                << "\nL1GlobalTriggerRecord found in the event with \n  "
                << l1GtRecordInputTag << std::endl;

    }

    if (foundL1GtReadoutRecord) {
        edm::LogVerbatim("L1GtUtils")
                << "\nL1GlobalTriggerReadoutRecord found in the event with \n  "
                << l1GtReadoutRecordInputTag << std::endl;
    }

}

bool L1GtUtils::l1AlgTechTrigBitNumber(const std::string& nameAlgTechTrig,
        int& triggerAlgTechTrig, int& bitNumber) {

    triggerAlgTechTrig = -1;
    bitNumber = -1;

    // test if the name is an algorithm alias
    CItAlgo itAlgo = m_algorithmAliasMap->find(nameAlgTechTrig);
    if (itAlgo != m_algorithmAliasMap->end()) {
        triggerAlgTechTrig = 0;
        bitNumber = (itAlgo->second).algoBitNumber();

        return true;
    }

    // test if the name is an algorithm name
    itAlgo = m_algorithmMap->find(nameAlgTechTrig);
    if (itAlgo != m_algorithmMap->end()) {
        triggerAlgTechTrig = 0;
        bitNumber = (itAlgo->second).algoBitNumber();

        return true;
    }

    // test if the name is a technical trigger
    itAlgo = m_technicalTriggerMap->find(nameAlgTechTrig);
    if (itAlgo != m_technicalTriggerMap->end()) {
        triggerAlgTechTrig = 1;
        bitNumber = (itAlgo->second).algoBitNumber();

        return true;
    }

    return false;

}


int L1GtUtils::l1Results(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, bool& decisionBeforeMask,
        bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) {

    // initial values for returned results
    decisionBeforeMask = false;
    decisionAfterMask = false;
    prescaleFactor = -1;
    triggerMask = -1;

    // initialize error code
    int iError = 0;

    // if the given name is not an physics algorithm alias, a physics algorithm name
    // or a technical trigger in the current menu, return with error code 1
    int triggerAlgTechTrig = -1;
    int bitNumber = -1;


    if (!l1AlgTechTrigBitNumber(nameAlgTechTrig, triggerAlgTechTrig,
            bitNumber)) {

        iError = iError + 1;
        LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n"
                << nameAlgTechTrig << " not found in the trigger menu \n"
                << m_l1GtMenu->gtTriggerMenuImplementation() << std::endl;

        return iError;

    }

    if (bitNumber < 0) {

        iError = iError + 2;
        LogDebug("L1GtUtils")
                << "\nBit number for algorithm/technical trigger \n"
                << nameAlgTechTrig << " from menu \n"
                << m_l1GtMenu->gtTriggerMenuImplementation() << " negative. "
                << std::endl;

        return iError;
    }


    // intermediate error code for the records
    // the module returns an error code only if both the lite and the readout record are missing
    int iErrorRecord = 0;

    // get L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
    // in L1GlobalTriggerRecord, only the physics partition is available
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    edm::Handle<L1GlobalTriggerRecord> gtRecord;

    iEvent.getByLabel(l1GtReadoutRecordInputTag, gtReadoutRecord);
    iEvent.getByLabel(l1GtRecordInputTag, gtRecord);

    bool validRecord = false;

    // initialization, update is done later from the record
    unsigned int pfIndexAlgTechTrig = 0;

    if (gtRecord.isValid()) {


        if (triggerAlgTechTrig) {
            pfIndexAlgTechTrig = gtRecord->gtPrescaleFactorIndexTech();
        } else {
            pfIndexAlgTechTrig = gtRecord->gtPrescaleFactorIndexAlgo();
        }

        validRecord = true;

    } else {

        iErrorRecord = 10;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nnot found in the event." << std::endl;

    }

    if (gtReadoutRecord.isValid()) {

        if (triggerAlgTechTrig) {
            pfIndexAlgTechTrig =
                    static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexTech());
        } else {
            pfIndexAlgTechTrig =
                    static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo());
        }

        validRecord = true;

    } else {

        iErrorRecord = iErrorRecord + 100;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerReadoutRecord with \n  "
                << l1GtRecordInputTag << "\nnot found in the event." << std::endl;

    }

    if (!validRecord) {

        LogDebug("L1GtUtils") << "\nError: "
                << "\nNo valid L1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nfound."
                << "\nNo valid L1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nfound in the event." << std::endl;

        iError = iErrorRecord;
        return iError;
    }

    // get the prescale factor set used in the actual luminosity segment
    // first, check if a correct index is retrieved

    size_t pfSetsSize = 0;

    if (triggerAlgTechTrig) {
        pfSetsSize = m_prescaleFactorsTechTrig->size();
    } else {
        pfSetsSize = m_prescaleFactorsAlgoTrig->size();
    }


    if (pfIndexAlgTechTrig < 0) {

        iError = iError + 1000;
        LogDebug("L1GtUtils")
                << "\nIndex of prescale factor set retrieved from the data \n"
                << "less than zero.\n" << "  Index from data = "
                << pfIndexAlgTechTrig << std::endl;

        return iError;

    } else if (pfIndexAlgTechTrig >= pfSetsSize) {
        iError = iError + 2000;
        LogDebug("L1GtUtils")
                << "\nIndex of prescale factor set retrieved from the data \n"
                << "greater than the size of the vector of prescale factor sets.\n"
                << "  Index from data = " << pfIndexAlgTechTrig << "  Size = "
                << pfSetsSize << std::endl;

        return iError;

    }

    const std::vector<int>* prescaleFactorsAlgTechTrig = 0;

    if (triggerAlgTechTrig) {
        prescaleFactorsAlgTechTrig = &((*m_prescaleFactorsTechTrig).at(pfIndexAlgTechTrig));
    } else {
        prescaleFactorsAlgTechTrig = &((*m_prescaleFactorsAlgoTrig).at(pfIndexAlgTechTrig));
    }



    // algorithm result before applying the trigger masks

    if (gtReadoutRecord.isValid()) {


        if (triggerAlgTechTrig) {
            // technical trigger
            const DecisionWord& gtDecisionWordBeforeMask =
                    gtReadoutRecord->technicalTriggerWord();

            if (bitNumber < static_cast<int>(gtDecisionWordBeforeMask.size())) {
                decisionBeforeMask = gtDecisionWordBeforeMask[bitNumber];
            } else {
                iError = iError + 3000;
                LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                        << " for technical trigger \n" << nameAlgTechTrig
                        << " greater than size of L1 GT decision word: "
                        << gtDecisionWordBeforeMask.size()
                        << "\nError: Inconsistent L1 trigger event setup!"
                        << std::endl;

                return iError;

            }

        } else {
            // physics algorithm
            const DecisionWord& gtDecisionWordBeforeMask =
                    gtReadoutRecord->decisionWord();

            if (bitNumber < static_cast<int>(gtDecisionWordBeforeMask.size())) {
                decisionBeforeMask = gtDecisionWordBeforeMask[bitNumber];
            } else {
                iError = iError + 3000;
                LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                        << " for physics algorithm \n" << nameAlgTechTrig
                        << " greater than size of L1 GT decision word: "
                        << gtDecisionWordBeforeMask.size()
                        << "\nError: Inconsistent L1 trigger event setup!"
                        << std::endl;

                return iError;

            }
        }

    } else if (gtRecord.isValid()) {

        if (triggerAlgTechTrig) {
            // technical trigger
            const DecisionWord& gtDecisionWordBeforeMask =
                    gtRecord->technicalTriggerWordBeforeMask();

            if (bitNumber < static_cast<int>(gtDecisionWordBeforeMask.size())) {
                decisionBeforeMask = gtDecisionWordBeforeMask[bitNumber];
            } else {
                iError = iError + 3000;
                LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                        << " for technical trigger \n" << nameAlgTechTrig
                        << " greater than size of L1 GT decision word: "
                        << gtDecisionWordBeforeMask.size()
                        << "\nError: Inconsistent L1 trigger event setup!"
                        << std::endl;

                return iError;

            }

        } else {
            // physics algorithm
            const DecisionWord& gtDecisionWordBeforeMask =
                    gtRecord->decisionWordBeforeMask();

            if (bitNumber < static_cast<int>(gtDecisionWordBeforeMask.size())) {
                decisionBeforeMask = gtDecisionWordBeforeMask[bitNumber];
            } else {
                iError = iError + 3000;
                LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                        << " for physics algorithm \n" << nameAlgTechTrig
                        << " greater than size of L1 GT decision word: "
                        << gtDecisionWordBeforeMask.size()
                        << "\nError: Inconsistent L1 trigger event setup!"
                        << std::endl;

                return iError;

            }
        }

    }


    // prescale factor
    if (bitNumber < static_cast<int>(prescaleFactorsAlgTechTrig->size())) {
        prescaleFactor = (*prescaleFactorsAlgTechTrig)[bitNumber];
    } else {
        iError = iError + 4000;
        LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                << " for algorithm/technical trigger \n" << nameAlgTechTrig
                << " negative or greater than size of L1 GT prescale factor vector set: "
                << prescaleFactorsAlgTechTrig->size()
                << "\nError: Inconsistent L1 trigger event setup!" << std::endl;

        return iError;

    }

    // algorithm result after applying the trigger masks
    if (bitNumber < static_cast<int>(m_triggerMaskAlgoTrig.size())) {
        triggerMask = (m_triggerMaskAlgoTrig[bitNumber]) & (1
                << m_physicsDaqPartition);
    } else {
        iError = iError + 5000;
        LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                << " for algorithm/technical trigger \n" << nameAlgTechTrig
                << " negative or greater than size of L1 GT trigger mask set: "
                << m_triggerMaskAlgoTrig.size()
                << "\nError: Inconsistent L1 trigger event setup!" << std::endl;

        return iError;

    }

    decisionAfterMask = decisionBeforeMask;

    if (triggerMask) {
        decisionAfterMask = false;
    }

    return iError;

}


int L1GtUtils::l1Results(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, bool& decisionBeforeMask,
        bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) {

    edm::InputTag l1GtRecordInputTag;
    edm::InputTag l1GtReadoutRecordInputTag;

    // initial values for returned results
    decisionBeforeMask = false;
    decisionAfterMask = false;
    prescaleFactor = -1;
    triggerMask = -1;

    getInputTag(iEvent, l1GtRecordInputTag, l1GtReadoutRecordInputTag);

    int l1ErrorCode = 0;

    l1ErrorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return l1ErrorCode;

}

//

const bool L1GtUtils::decisionBeforeMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionBeforeMask;

}

const bool L1GtUtils::decisionBeforeMask(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionBeforeMask;

}

//

const bool L1GtUtils::decisionAfterMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

const bool L1GtUtils::decisionAfterMask(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

//

const bool L1GtUtils::decision(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

const bool L1GtUtils::decision(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

//

const int L1GtUtils::prescaleFactor(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return prescaleFactor;

}

const int L1GtUtils::prescaleFactor(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return prescaleFactor;

}

const int L1GtUtils::triggerMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return triggerMask;

}

const int L1GtUtils::triggerMask(const edm::Event& iEvent,
        const std::string& nameAlgTechTrig, int& errorCode) {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return triggerMask;

}

const int L1GtUtils::triggerMask(const std::string& nameAlgTechTrig,
        int& errorCode) {

    // initial values for returned results
    int triggerMaskValue = -1;

    // initialize error code
    int iError = 0;

    // if the given name is not an physics algorithm alias, a physics algorithm name
    // or a technical trigger in the current menu, return with error code 1
    int triggerAlgTechTrig = -1;
    int bitNumber = -1;

    if (!l1AlgTechTrigBitNumber(nameAlgTechTrig, triggerAlgTechTrig,
            bitNumber)) {

        iError = iError + 1;
        LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n"
                << nameAlgTechTrig << " not found in the trigger menu \n"
                << m_l1GtMenu->gtTriggerMenuImplementation() << std::endl;

        errorCode = iError;
        return triggerMaskValue;

    }


    if (bitNumber < 0) {
        iError = iError + 2;
        LogDebug("L1GtUtils")
                << "\nBit number for algorithm/technical trigger \n"
                << nameAlgTechTrig << " from menu \n"
                << m_l1GtMenu->gtTriggerMenuImplementation() << " negative. "
                << std::endl;

        return iError;
    }


    if (bitNumber < static_cast<int>(m_triggerMaskAlgoTrig.size())) {
        triggerMaskValue =  (m_triggerMaskAlgoTrig[bitNumber]) & (1
                << m_physicsDaqPartition);

    } else {
        iError = iError + 5000;
        LogDebug("L1GtUtils") << "\nBit number " << bitNumber
                << " for algorithm/technical trigger \n" << nameAlgTechTrig
                << " greater than size of L1 GT trigger mask set: "
                << m_triggerMaskAlgoTrig.size()
                << "\nError: Inconsistent L1 trigger event setup!" << std::endl;

        errorCode = iError;
        return triggerMaskValue;

    }

    errorCode = iError;
    return triggerMaskValue;

}

const std::vector<int>& L1GtUtils::prescaleFactorSet(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& triggerAlgTechTrig, int& errorCode) {

    // test if the argument for the "trigger algorithm type" is correct
    if ((triggerAlgTechTrig == "TechnicalTriggers") || ((triggerAlgTechTrig
            == "PhysicsAlgorithms"))) {

        LogDebug("L1GtUtils")
                << "\nPrescale factor set to be retrieved for the argument "
                << triggerAlgTechTrig << std::endl;
    } else {

        LogDebug("L1GtUtils")
                << "\nPrescale factor set cannot be retrieved for the argument "
                << triggerAlgTechTrig
                << "\n  Supported arguments: 'PhysicsAlgorithms' or 'TechnicalTriggers'"
                << std::endl;

        errorCode = 6000;
        return m_prescaleFactorSet;

    }

    // initialize error code
    int iError = 0;

    // intermediate error code for the records
    // the module returns an error code only if both the lite and the readout record are missing
    int iErrorRecord = 0;

    // get L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
    // in L1GlobalTriggerRecord, only the physics partition is available
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    edm::Handle<L1GlobalTriggerRecord> gtRecord;

    iEvent.getByLabel(l1GtRecordInputTag, gtRecord);
    iEvent.getByLabel(l1GtReadoutRecordInputTag, gtReadoutRecord);

    bool validRecord = false;

    // initialization, update is done later from the record
    unsigned int pfIndexAlgTechTrig = 0;

    if (gtRecord.isValid()) {


        if (triggerAlgTechTrig == "TechnicalTriggers") {
            pfIndexAlgTechTrig = gtRecord->gtPrescaleFactorIndexTech();
        } else {
            pfIndexAlgTechTrig = gtRecord->gtPrescaleFactorIndexAlgo();
        }

        validRecord = true;

    } else {

        iErrorRecord = 10;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nnot found" << std::endl;

    }

    if (gtReadoutRecord.isValid()) {

        if (triggerAlgTechTrig == "TechnicalTriggers") {
            pfIndexAlgTechTrig =
                    static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexTech());
        } else {
            pfIndexAlgTechTrig =
                    static_cast<unsigned int> ((gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo());
        }

        validRecord = true;

    } else {

        iErrorRecord = iErrorRecord + 100;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerReadoutRecord with \n  "
                << l1GtRecordInputTag << "\nnot found" << std::endl;

    }

    if (!validRecord) {

        LogDebug("L1GtUtils") << "\nError: "
                << "\nNo valid L1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nfound."
                << "\nNo valid L1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nfound." << std::endl;

        errorCode = iErrorRecord;
        return m_prescaleFactorSet;
    }

    // get the prescale factor set used in the actual luminosity segment
    // first, check if a correct index is retrieved

    size_t pfSetsSize = 0;

    if (triggerAlgTechTrig == "TechnicalTriggers") {
        pfSetsSize = m_prescaleFactorsTechTrig->size();
    } else {
        pfSetsSize = m_prescaleFactorsAlgoTrig->size();
    }

    if (pfIndexAlgTechTrig < 0) {

        iError = iError + 1000;
        LogDebug("L1GtUtils")
                << "\nIndex of prescale factor set retrieved from the data \n"
                << "less than zero.\n" << "  Index from data = "
                << pfIndexAlgTechTrig << std::endl;

        errorCode = iError;
        return m_prescaleFactorSet;

    } else if (pfIndexAlgTechTrig >= pfSetsSize) {
        iError = iError + 2000;
        LogDebug("L1GtUtils")
                << "\nIndex of prescale factor set retrieved from the data \n"
                << "greater than the size of the vector of prescale factor sets.\n"
                << "  Index from data = " << pfIndexAlgTechTrig << "  Size = "
                << pfSetsSize << std::endl;

        errorCode = iError;
        return m_prescaleFactorSet;

    }

    if (triggerAlgTechTrig == "TechnicalTriggers") {
        m_prescaleFactorSet = (*m_prescaleFactorsTechTrig)[pfIndexAlgTechTrig];
    } else {
        m_prescaleFactorSet = (*m_prescaleFactorsAlgoTrig)[pfIndexAlgTechTrig];
    }

    errorCode = iError;
    return m_prescaleFactorSet;

}

const std::vector<int>& L1GtUtils::prescaleFactorSet(const edm::Event& iEvent,
        const std::string& triggerAlgTechTrig, int& errorCode) {

    // initialize error code
    int iError = 0;

    edm::InputTag l1GtRecordInputTag;
    edm::InputTag l1GtReadoutRecordInputTag;

    getInputTag(iEvent, l1GtRecordInputTag, l1GtReadoutRecordInputTag);

    m_prescaleFactorSet = prescaleFactorSet(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, triggerAlgTechTrig, iError);

    errorCode = iError;
    return m_prescaleFactorSet;

}



const std::vector<unsigned int>& L1GtUtils::triggerMaskSet(
        const std::string& triggerAlgTechTrig, int& errorCode) {

    // clear the vector before filling it with push_back
    m_triggerMaskSet.clear();

    // initialize error code
    int iError = 0;

    if (triggerAlgTechTrig == "PhysicsAlgorithms") {

        for (unsigned i = 0; i < m_triggerMaskAlgoTrig.size(); i++) {
            m_triggerMaskSet.push_back((m_triggerMaskAlgoTrig[i]) & (1
                    << m_physicsDaqPartition));
        }

    } else if (triggerAlgTechTrig == "TechnicalTriggers") {

        for (unsigned i = 0; i < m_triggerMaskTechTrig.size(); i++) {
            m_triggerMaskSet.push_back((m_triggerMaskTechTrig[i]) & (1
                    << m_physicsDaqPartition));
        }
    } else {

        iError = iError + 6000;
        LogDebug("L1GtUtils")
                << "\nTrigger mask  cannot be retrieved for the argument "
                << triggerAlgTechTrig
                << "\n  Supported arguments: 'PhysicsAlgorithms' or 'TechnicalTriggers'"
                << std::endl;

    }

    errorCode = iError;
    return m_triggerMaskSet;

}

const std::string L1GtUtils::l1TriggerMenu() {

    return m_l1GtMenu->gtTriggerMenuImplementation();

}

