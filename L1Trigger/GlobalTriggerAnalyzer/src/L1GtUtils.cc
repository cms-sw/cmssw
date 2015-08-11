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
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// system include files
#include <iomanip>
#include <vector>

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

    m_l1GtStableParCacheID(0ULL), m_numberAlgorithmTriggers(0),

    m_numberTechnicalTriggers(0),

    m_l1GtPfAlgoCacheID(0ULL), m_l1GtPfTechCacheID(0ULL),

    m_l1GtTmAlgoCacheID(0ULL), m_l1GtTmTechCacheID(0ULL),

    m_l1GtTmVetoAlgoCacheID(0ULL), m_l1GtTmVetoTechCacheID(0ULL),

    m_l1GtMenuCacheID(0ULL),

    m_l1EventSetupValid(false),

    m_l1GtMenuLiteValid(false),

    m_beginRunCache(false),

    m_runIDCache(0),

    m_provRunIDCache(0),
    
    m_physicsDaqPartition(0),

    m_retrieveL1EventSetup(false),

    m_retrieveL1GtTriggerMenuLite(false)

    {

    // empty
}

// destructor
L1GtUtils::~L1GtUtils() {

    // empty

}

const std::string L1GtUtils::triggerCategory(
        const TriggerCategory& trigCategory) const {

    switch (trigCategory) {
        case AlgorithmTrigger: {
            return "Algorithm Trigger";
        }
            break;
        case TechnicalTrigger: {
            return "Technical Trigger";
        }

            break;
        default: {
            return EmptyString;
        }
            break;
    }
}


void L1GtUtils::retrieveL1EventSetup(const edm::EventSetup& evSetup) {

    //
    m_retrieveL1EventSetup = true;

    m_l1EventSetupValid = true;
    // FIXME test for each record if valid; if not set m_l1EventSetupValid = false;

    // get / update the stable parameters from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID =
            evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {

        edm::ESHandle<L1GtStableParameters> l1GtStablePar;
        evSetup.get<L1GtStableParametersRcd>().get(l1GtStablePar);
        m_l1GtStablePar = l1GtStablePar.product();

        // number of algorithm triggers
        m_numberAlgorithmTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of technical triggers
        m_numberTechnicalTriggers =
                m_l1GtStablePar->gtNumberTechnicalTriggers();

        int maxNumberTrigger = std::max(m_numberAlgorithmTriggers,
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

        m_triggerMaskAlgoTrig = &(m_l1GtTmAlgo->gtTriggerMask());

        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }

    unsigned long long l1GtTmTechCacheID =
            evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmTech;
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().get(l1GtTmTech);
        m_l1GtTmTech = l1GtTmTech.product();

        m_triggerMaskTechTrig = &(m_l1GtTmTech->gtTriggerMask());

        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }

    unsigned long long l1GtTmVetoAlgoCacheID =
            evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoAlgo;
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().get(l1GtTmVetoAlgo);
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

        m_triggerMaskVetoAlgoTrig = &(m_l1GtTmVetoAlgo->gtTriggerMask());

        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }

    unsigned long long l1GtTmVetoTechCacheID =
            evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoTechCacheID != l1GtTmVetoTechCacheID) {

        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoTech;
        evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().get(l1GtTmVetoTech);
        m_l1GtTmVetoTech = l1GtTmVetoTech.product();

        m_triggerMaskVetoTechTrig = &(m_l1GtTmVetoTech->gtTriggerMask());

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


void L1GtUtils::retrieveL1GtTriggerMenuLite(const edm::Run& iRun,
        const edm::InputTag& l1GtMenuLiteInputTag) {

    //
    m_retrieveL1GtTriggerMenuLite = true;

    // get L1GtTriggerMenuLite
    edm::Handle<L1GtTriggerMenuLite> l1GtMenuLite;
    iRun.getByLabel(l1GtMenuLiteInputTag, l1GtMenuLite);

    if (!l1GtMenuLite.isValid()) {

        LogDebug("L1GtUtils") << "\nL1GtTriggerMenuLite with \n  "
                << l1GtMenuLiteInputTag
                << "\nrequested in configuration, but not found in the event."
                << std::endl;

        m_l1GtMenuLiteValid = false;
    } else {
        m_l1GtMenuLite = l1GtMenuLite.product();
        m_l1GtMenuLiteValid = true;

        LogDebug("L1GtUtils") << "\nL1GtTriggerMenuLite with \n  "
                << l1GtMenuLiteInputTag << "\nretrieved for run "
                << iRun.runAuxiliary().run() << std::endl;

        m_algorithmMapLite = &(m_l1GtMenuLite->gtAlgorithmMap());
        m_algorithmAliasMapLite = &(m_l1GtMenuLite->gtAlgorithmAliasMap());
        m_technicalTriggerMapLite = &(m_l1GtMenuLite->gtTechnicalTriggerMap());

        m_triggerMaskAlgoTrigLite = &(m_l1GtMenuLite->gtTriggerMaskAlgoTrig());
        m_triggerMaskTechTrigLite = &(m_l1GtMenuLite->gtTriggerMaskTechTrig());

        m_prescaleFactorsAlgoTrigLite
                = &(m_l1GtMenuLite->gtPrescaleFactorsAlgoTrig());
        m_prescaleFactorsTechTrigLite
                = &(m_l1GtMenuLite->gtPrescaleFactorsTechTrig());

    }

}

void L1GtUtils::getL1GtRunCache(const edm::Run& iRun,
        const edm::EventSetup& evSetup, const bool useL1EventSetup,
        const bool useL1GtTriggerMenuLite, const edm::InputTag& l1GtTmLInputTag) {

    // first call will turn this to true: the quantities which can be cached in
    // beginRun will not be cached then in analyze
    m_beginRunCache = true;

    // if requested, retrieve and cache L1 event setup
    // keep the caching based on cacheIdentifier() for each record
    if (useL1EventSetup) {
        retrieveL1EventSetup(evSetup);
    }

    // cached per run

    // if requested, retrieve and cache the L1GtTriggerMenuLite
    // L1GtTriggerMenuLite is defined per run and produced in prompt reco by L1Reco
    // and put in the Run section
    if (useL1GtTriggerMenuLite) {
        retrieveL1GtTriggerMenuLite(iRun, l1GtTmLInputTag);
    }

}


void L1GtUtils::getL1GtRunCache(const edm::Run& iRun,
        const edm::EventSetup& evSetup, bool useL1EventSetup,
        bool useL1GtTriggerMenuLite) {

    if (useL1GtTriggerMenuLite) {
        getL1GtTriggerMenuLiteInputTag(iRun, m_provL1GtTriggerMenuLiteInputTag);

    }

    getL1GtRunCache(iRun, evSetup, useL1EventSetup, useL1GtTriggerMenuLite,
            m_provL1GtTriggerMenuLiteInputTag);

}



void L1GtUtils::getL1GtRunCache(const edm::Event& iEvent,
        const edm::EventSetup& evSetup, const bool useL1EventSetup,
        const bool useL1GtTriggerMenuLite, const edm::InputTag& l1GtTmLInputTag) {

    // if there was no retrieval and caching in beginRun, do it here
    if (!m_beginRunCache) {

        // if requested, retrieve and cache L1 event setup
        // keep the caching based on cacheIdentifier() for each record
        if (useL1EventSetup) {
            retrieveL1EventSetup(evSetup);
        }
    }

    // cached per run

    const edm::Run& iRun = iEvent.getRun();
    edm::RunID runID = iRun.runAuxiliary().id();

    if (runID != m_runIDCache) {

        if (!m_beginRunCache) {
            // if requested, retrieve and cache the L1GtTriggerMenuLite
            // L1GtTriggerMenuLite is defined per run and produced in prompt reco by L1Reco
            // and put in the Run section
            if (useL1GtTriggerMenuLite) {
                retrieveL1GtTriggerMenuLite(iRun, l1GtTmLInputTag);
            }
        }

        // find from provenance and cache the input tags for L1GlobalTriggerRecord and
        // L1GlobalTriggerReadoutRecord
        getL1GtRecordInputTag(iEvent, m_provL1GtRecordInputTag,
                m_provL1GtReadoutRecordInputTag);

        //
        m_runIDCache = runID;

    }

}


void L1GtUtils::getL1GtRunCache(const edm::Event& iEvent,
        const edm::EventSetup& evSetup, const bool useL1EventSetup,
        const bool useL1GtTriggerMenuLite) {

    // if the input tag for L1GtTriggerMenuLite was not found in beginRun, do it here
    if (!m_beginRunCache) {

        const edm::Run& iRun = iEvent.getRun();
        edm::RunID runID = iRun.runAuxiliary().id();

        if (runID != m_provRunIDCache) {

            if (useL1GtTriggerMenuLite) {

                getL1GtTriggerMenuLiteInputTag(iRun,
                        m_provL1GtTriggerMenuLiteInputTag);
            }

            //
            m_provRunIDCache = runID;
        }

    }

    // call now the general method for getL1GtRunCache
    getL1GtRunCache(iEvent, evSetup, useL1EventSetup, useL1GtTriggerMenuLite,
            m_provL1GtTriggerMenuLiteInputTag);

}


void L1GtUtils::getL1GtRecordInputTag(const edm::Event& iEvent,
        edm::InputTag& l1GtRecordInputTag,
        edm::InputTag& l1GtReadoutRecordInputTag) const {

    typedef std::vector<edm::Provenance const*> Provenances;
    Provenances provenances;
    std::string friendlyName;
    std::string modLabel;
    std::string instanceName;
    std::string processName;

    bool foundL1GtRecord = false;
    bool foundL1GtReadoutRecord = false;

    bool foundMultipleL1GtRecord = false;
    bool foundMultipleL1GtReadoutRecord = false;

    // use vector here, InputTag has no '<' operator to use std::set
    std::vector<edm::InputTag> inputTagsL1GtRecord;
    std::vector<edm::InputTag> inputTagsL1GtReadoutRecord;

    // TODO decide if the preferred input tags must be given as input parameters
    // or stay hardwired
    std::vector<edm::InputTag> preferredL1GtRecordInputTag;
    preferredL1GtRecordInputTag.reserve(5);
    preferredL1GtRecordInputTag.push_back(edm::InputTag("gtDigis"));
    preferredL1GtRecordInputTag.push_back(edm::InputTag("hltGtDigis"));

    std::vector<edm::InputTag> preferredL1GtReadoutRecordInputTag;
    preferredL1GtReadoutRecordInputTag.reserve(5);
    preferredL1GtReadoutRecordInputTag.push_back(edm::InputTag("gtDigis"));
    preferredL1GtReadoutRecordInputTag.push_back(edm::InputTag("hltGtDigis"));

    LogDebug("L1GtUtils") << "\nTry to get AllProvenance for event "
            << iEvent.id().event() << std::endl;

    iEvent.getAllProvenance(provenances);

    LogTrace("L1GtUtils") << "\n" << "Event contains " << provenances.size()
            << " product" << (provenances.size() == 1 ? "" : "s")
            << " with friendlyClassName, moduleLabel, productInstanceName and processName:\n"
            << std::endl;

    for (Provenances::iterator itProv = provenances.begin(), itProvEnd =
            provenances.end(); itProv != itProvEnd; ++itProv) {

        friendlyName = (*itProv)->friendlyClassName();
        modLabel = (*itProv)->moduleLabel();
        instanceName = (*itProv)->productInstanceName();
        processName = (*itProv)->processName();

        LogTrace("L1GtUtils") << friendlyName << "\t \"" << modLabel
                << "\" \t \"" << instanceName << "\" \t \"" << processName
                << "\"" << std::endl;

        if (friendlyName == "L1GlobalTriggerRecord") {
            l1GtRecordInputTag = edm::InputTag(modLabel, instanceName,
                    processName);
            inputTagsL1GtRecord.push_back(l1GtRecordInputTag);

            foundL1GtRecord = true;
            if (inputTagsL1GtRecord.size() > 1) {
                foundMultipleL1GtRecord = true;
            }
        } else if (friendlyName == "L1GlobalTriggerReadoutRecord") {
            l1GtReadoutRecordInputTag = edm::InputTag(modLabel, instanceName,
                    processName);
            inputTagsL1GtReadoutRecord.push_back(l1GtReadoutRecordInputTag);

            foundL1GtReadoutRecord = true;

             if (inputTagsL1GtReadoutRecord.size() > 1) {
                foundMultipleL1GtReadoutRecord = true;
            }

        }
    }

    // print statistics of L1GlobalTriggerReadoutRecord input tags
    int nrInpuTags = inputTagsL1GtReadoutRecord.size();
    if (nrInpuTags == 1) {
        LogDebug("L1GtUtils") << "\n Unique L1GlobalTriggerReadoutRecord product found in the event."
                << std::endl;

    } else if (nrInpuTags > 1) {
        LogDebug("L1GtUtils") << "\nWARNING: Multiple L1GlobalTriggerReadoutRecord products found in the event.\n  "
                << nrInpuTags
                << " L1GlobalTriggerReadoutRecord products:"
                << std::endl;
    } else {
        LogDebug("L1GtUtils") << "\nWARNING: No L1GlobalTriggerReadoutRecord product found in the event.\n  "
                << std::endl;
    }

    for (int iTag = 0; iTag < nrInpuTags; ++iTag) {
        edm::LogPrint("L1GtUtils")
                << inputTagsL1GtReadoutRecord[iTag]
                << std::endl;
    }

    // if L1GlobalTriggerRecord and L1GlobalTriggerReadoutRecord are not found, return empty input tags
    // if multiple tags are found, write a warning and check if a "preferred input tag" is
    // among the tags - if true, take this tag, otherwise return an empty tag.
    if (!foundL1GtRecord) {
        l1GtRecordInputTag = edm::InputTag();
    } else if (foundMultipleL1GtRecord)  {

        bool foundPreferredInputTag = false;
        for (std::vector<edm::InputTag>::const_iterator itTag = inputTagsL1GtRecord.begin(), itTagEnd =
                inputTagsL1GtRecord.end(); itTag != itTagEnd; ++itTag) {

            for (std::vector<edm::InputTag>::const_iterator itPrefTag = preferredL1GtRecordInputTag.begin(), itPrefTagEnd =
                    preferredL1GtRecordInputTag.end(); itPrefTag != itPrefTagEnd; ++itPrefTag) {

                    if (itTag->label() == itPrefTag->label()) {
                        LogDebug("L1GtUtils")
                            << "\nWARNING: multiple L1GlobalTriggerRecord products found in the event."
                            << "\nInput tag set to the preferred input tag" << (*itTag)
                            << std::endl;
                        l1GtRecordInputTag = *itTag;
                        foundPreferredInputTag = true;
                        break;
                    }
            }

            if (foundPreferredInputTag) {
                break;
            }
        }

        if (!foundPreferredInputTag) {
            LogDebug("L1GtUtils")
                    << "\nWARNING: multiple L1GlobalTriggerRecord products found in the event."
                    << "\nNo preferred input tag within the found values"
                    << "\n Input tag put to empty input tag."
                    << std::endl;
            l1GtRecordInputTag = edm::InputTag();
        }

    } else {
        LogTrace("L1GtUtils")
                << "\nUnique L1GlobalTriggerRecord found in the event with \n  "
                << l1GtRecordInputTag << std::endl;
    }

    if (!foundL1GtReadoutRecord) {
        l1GtReadoutRecordInputTag = edm::InputTag();
    } else if (foundMultipleL1GtReadoutRecord)  {

        bool foundPreferredInputTag = false;
        for (std::vector<edm::InputTag>::const_iterator itTag = inputTagsL1GtReadoutRecord.begin(), itTagEnd =
                inputTagsL1GtReadoutRecord.end(); itTag != itTagEnd; ++itTag) {

            for (std::vector<edm::InputTag>::const_iterator itPrefTag = preferredL1GtReadoutRecordInputTag.begin(), itPrefTagEnd =
                    preferredL1GtReadoutRecordInputTag.end(); itPrefTag != itPrefTagEnd; ++itPrefTag) {

                    if (itTag->label() == itPrefTag->label()) {
                        LogDebug("L1GtUtils")
                        << "\nWARNING: multiple L1GlobalTriggerReadoutRecord products found in the event."
                        << "\nInput tag set to the preferred input tag" << (*itTag)
                        << std::endl;
                        l1GtReadoutRecordInputTag = *itTag;
                        foundPreferredInputTag = true;
                        break;
                    }
            }

            if (foundPreferredInputTag) {
                break;
            }
        }

        if (!foundPreferredInputTag) {
            LogDebug("L1GtUtils")
                    << "\nWARNING: multiple L1GlobalTriggerReadoutRecord products found in the event."
                    << "\nNo preferred input tag within the found values"
                    << "\n Input tag put to empty input tag."
                    << std::endl;
            l1GtReadoutRecordInputTag = edm::InputTag();
        }

    } else {
        LogTrace("L1GtUtils")
                << "\nUnique L1GlobalTriggerReadoutRecord found in the event with \n  "
                << l1GtReadoutRecordInputTag << std::endl;
    }

}

void L1GtUtils::getL1GtTriggerMenuLiteInputTag(const edm::Run& iRun,
        edm::InputTag& l1GtTriggerMenuLiteInputTag) const {

    typedef std::vector<edm::Provenance const*> Provenances;
    Provenances provenances;
    std::string friendlyName;
    std::string modLabel;
    std::string instanceName;
    std::string processName;

    bool foundL1GtTriggerMenuLite = false;

    LogDebug("L1GtUtils") << "\nTry to get AllProvenance for run "
            << iRun.runAuxiliary().run() << std::endl;

    iRun.getAllProvenance(provenances);

    LogTrace("L1GtUtils") << "\n" << "Run contains " << provenances.size()
            << " product" << (provenances.size() == 1 ? "" : "s")
            << " with friendlyClassName, moduleLabel, productInstanceName and processName:\n"
            << std::endl;

    for (Provenances::iterator itProv = provenances.begin(), itProvEnd =
            provenances.end(); itProv != itProvEnd; ++itProv) {

        friendlyName = (*itProv)->friendlyClassName();
        modLabel = (*itProv)->moduleLabel();
        instanceName = (*itProv)->productInstanceName();
        processName = (*itProv)->processName();

        LogTrace("L1GtUtils") << friendlyName << "\t \"" << modLabel
                << "\" \t \"" << instanceName << "\" \t \"" << processName
                << "\"" << std::endl;

        if (friendlyName == "L1GtTriggerMenuLite") {
            l1GtTriggerMenuLiteInputTag = edm::InputTag(modLabel, instanceName,
                    processName);
            foundL1GtTriggerMenuLite = true;
        }

    }

    if (!foundL1GtTriggerMenuLite) {
        l1GtTriggerMenuLiteInputTag = edm::InputTag();
        LogTrace("L1GtUtils") << "\nL1GtTriggerMenuLite not found in Run"
                << std::endl;
    } else {
        LogTrace("L1GtUtils") << "\nL1GtTriggerMenuLite found in Run with \n  "
                << l1GtTriggerMenuLiteInputTag << std::endl;
    }

}


const bool L1GtUtils::l1AlgoTechTrigBitNumber(
        const std::string& nameAlgoTechTrig, TriggerCategory& trigCategory,
        int& bitNumber) const {

    trigCategory = AlgorithmTrigger;
    bitNumber = -1;

    if (m_retrieveL1GtTriggerMenuLite) {
        if (m_l1GtMenuLiteValid) {

            // test if the name is an algorithm alias
            for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                    m_algorithmAliasMapLite->begin(); itTrig
                    != m_algorithmAliasMapLite->end(); itTrig++) {

                if (itTrig->second == nameAlgoTechTrig) {

                    trigCategory = AlgorithmTrigger;
                    bitNumber = itTrig->first;

                    return true;
                }
            }

            // test if the name is an algorithm name
            for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                    m_algorithmMapLite->begin(); itTrig
                    != m_algorithmMapLite->end(); itTrig++) {

                if (itTrig->second == nameAlgoTechTrig) {

                    trigCategory = AlgorithmTrigger;
                    bitNumber = itTrig->first;

                    return true;
                }
            }

            // test if the name is a technical trigger
            for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                    m_technicalTriggerMapLite->begin(); itTrig
                    != m_technicalTriggerMapLite->end(); itTrig++) {

                if (itTrig->second == nameAlgoTechTrig) {

                    trigCategory = TechnicalTrigger;
                    bitNumber = itTrig->first;

                    return true;
                }
            }

        } else if (m_retrieveL1EventSetup) {

            // test if the name is an algorithm alias
            CItAlgo itAlgo = m_algorithmAliasMap->find(nameAlgoTechTrig);
            if (itAlgo != m_algorithmAliasMap->end()) {
                trigCategory = AlgorithmTrigger;
                bitNumber = (itAlgo->second).algoBitNumber();

                return true;
            }

            // test if the name is an algorithm name
            itAlgo = m_algorithmMap->find(nameAlgoTechTrig);
            if (itAlgo != m_algorithmMap->end()) {
                trigCategory = AlgorithmTrigger;
                bitNumber = (itAlgo->second).algoBitNumber();

                return true;
            }

            // test if the name is a technical trigger
            itAlgo = m_technicalTriggerMap->find(nameAlgoTechTrig);
            if (itAlgo != m_technicalTriggerMap->end()) {
                trigCategory = TechnicalTrigger;
                bitNumber = (itAlgo->second).algoBitNumber();

                return true;
            }

        } else {
            // only L1GtTriggerMenuLite requested, but it is not valid
            return false;

        }
    } else if (m_retrieveL1EventSetup) {

        // test if the name is an algorithm alias
        CItAlgo itAlgo = m_algorithmAliasMap->find(nameAlgoTechTrig);
        if (itAlgo != m_algorithmAliasMap->end()) {
            trigCategory = AlgorithmTrigger;
            bitNumber = (itAlgo->second).algoBitNumber();

            return true;
        }

        // test if the name is an algorithm name
        itAlgo = m_algorithmMap->find(nameAlgoTechTrig);
        if (itAlgo != m_algorithmMap->end()) {
            trigCategory = AlgorithmTrigger;
            bitNumber = (itAlgo->second).algoBitNumber();

            return true;
        }

        // test if the name is a technical trigger
        itAlgo = m_technicalTriggerMap->find(nameAlgoTechTrig);
        if (itAlgo != m_technicalTriggerMap->end()) {
            trigCategory = TechnicalTrigger;
            bitNumber = (itAlgo->second).algoBitNumber();

            return true;
        }

    } else {
        // L1 trigger configuration not retrieved
        return false;

    }

    // all possibilities already tested, so it should not arrive here
    return false;


}

const bool L1GtUtils::l1TriggerNameFromBit(const int& bitNumber,
        const TriggerCategory& trigCategory, std::string& aliasL1Trigger,
        std::string& nameL1Trigger) const {

    aliasL1Trigger.clear();
    nameL1Trigger.clear();

    if (m_retrieveL1GtTriggerMenuLite) {
        if (m_l1GtMenuLiteValid) {

            // for an algorithm trigger
            if (trigCategory == AlgorithmTrigger) {

                bool trigAliasFound = false;
                bool trigNameFound = false;

                for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                        m_algorithmAliasMapLite->begin();
                        itTrig != m_algorithmAliasMapLite->end(); itTrig++) {

                    if (static_cast<int>(itTrig->first) == bitNumber) {
                        aliasL1Trigger = itTrig->second;
                        trigAliasFound = true;
                        break;
                    }
                }

                for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                        m_algorithmMapLite->begin();
                        itTrig != m_algorithmMapLite->end(); itTrig++) {

                    if (static_cast<int>(itTrig->first) == bitNumber) {
                        nameL1Trigger = itTrig->second;
                        trigNameFound = true;
                        break;
                    }
                }

                if (!(trigAliasFound && trigNameFound)) {
                    return false;
                }

                return true;

            } else if (trigCategory == TechnicalTrigger) {

                // for a technical trigger   

                bool trigNameFound = false;

                for (L1GtTriggerMenuLite::CItL1Trig itTrig =
                        m_technicalTriggerMapLite->begin();
                        itTrig != m_technicalTriggerMapLite->end(); itTrig++) {

                    if (static_cast<int>(itTrig->first) == bitNumber) {
                        nameL1Trigger = itTrig->second;

                        // technically, no alias is defined for technical triggers
                        // users use mainly aliases, so just return the name here
                        aliasL1Trigger = itTrig->second;

                        trigNameFound = true;
                        break;
                    }
                }

                if (!(trigNameFound)) {
                    return false;
                }

                return true;

            } else {

                // non-existing trigger category...
                return false;
            }

        } else if (m_retrieveL1EventSetup) {

            // for an algorithm trigger
            if (trigCategory == AlgorithmTrigger) {

                bool trigAliasFound = false;

                for (CItAlgo itTrig = m_algorithmAliasMap->begin();
                        itTrig != m_algorithmAliasMap->end(); itTrig++) {

                    if ((itTrig->second).algoBitNumber() == bitNumber) {
                        aliasL1Trigger = itTrig->first;
                        // get the name here, avoiding a loop on m_algorithmMap
                        nameL1Trigger = (itTrig->second).algoName();

                        trigAliasFound = true;
                        break;
                    }
                }

                if (!(trigAliasFound)) {
                    return false;
                }

                return true;

            } else if (trigCategory == TechnicalTrigger) {

                // for a technical trigger   

                bool trigNameFound = false;

                for (CItAlgo itTrig = m_technicalTriggerMap->begin();
                        itTrig != m_technicalTriggerMap->end(); itTrig++) {

                    if ((itTrig->second).algoBitNumber() == bitNumber) {
                        nameL1Trigger = (itTrig->second).algoName();
                        // technically, no alias is defined for technical triggers
                        // users use mainly aliases, so just return the name here
                        aliasL1Trigger = nameL1Trigger;

                        trigNameFound = true;
                        break;
                    }
                }

                if (!(trigNameFound)) {
                    return false;
                }

                return true;

            } else {

                // non-existing trigger category...
                return false;
            }

        } else {
            // only L1GtTriggerMenuLite requested, but it is not valid
            return false;

        }
    } else if (m_retrieveL1EventSetup) {

        // for an algorithm trigger
        if (trigCategory == AlgorithmTrigger) {

            bool trigAliasFound = false;

            for (CItAlgo itTrig = m_algorithmAliasMap->begin();
                    itTrig != m_algorithmAliasMap->end(); itTrig++) {

                if ((itTrig->second).algoBitNumber() == bitNumber) {
                    aliasL1Trigger = itTrig->first;
                    // get the name here, avoiding a loop on m_algorithmMap
                    nameL1Trigger = (itTrig->second).algoName();

                    trigAliasFound = true;
                    break;
                }
            }

            if (!(trigAliasFound)) {
                return false;
            }

            return true;

        } else if (trigCategory == TechnicalTrigger) {

            // for a technical trigger   

            bool trigNameFound = false;

            for (CItAlgo itTrig = m_technicalTriggerMap->begin();
                    itTrig != m_technicalTriggerMap->end(); itTrig++) {

                if ((itTrig->second).algoBitNumber() == bitNumber) {
                    nameL1Trigger = (itTrig->second).algoName();
                    // technically, no alias is defined for technical triggers
                    // users use mainly aliases, so just return the name here
                    aliasL1Trigger = itTrig->first;

                    trigNameFound = true;
                    break;
                }
            }

            if (!(trigNameFound)) {
                return false;
            }

            return true;

        } else {

            // non-existing trigger category...
            return false;
        }

    } else {
        // L1 trigger configuration not retrieved
        return false;

    }

    // all possibilities already tested, so it should not arrive here
    return false;

}

const int L1GtUtils::l1Results(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, bool& decisionBeforeMask,
        bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const {

    // initial values for returned results
    decisionBeforeMask = false;
    decisionAfterMask = false;
    prescaleFactor = -1;
    triggerMask = -1;

    // initialize error code and L1 configuration code
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        return iError;
    }

    // at this point, a valid L1 configuration is available, so the if/else if/else
    // can be simplified

    // if the given name is not an algorithm trigger alias, an algorithm trigger name
    // or a technical trigger in the current menu, return with error code 1

    TriggerCategory trigCategory = AlgorithmTrigger;
    int bitNumber = -1;


    if (!l1AlgoTechTrigBitNumber(nameAlgoTechTrig, trigCategory, bitNumber)) {

        iError = l1ConfCode + 1;

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {

                LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                        << nameAlgoTechTrig
                        << "\not found in the trigger menu \n  "
                        << m_l1GtMenuLite->gtTriggerMenuImplementation()
                        << "\nretrieved from L1GtTriggerMenuLite" << std::endl;

            } else {

                // fall through: L1 trigger configuration from event setup
                LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                        << nameAlgoTechTrig
                        << "\not found in the trigger menu \n  "
                        << m_l1GtMenu->gtTriggerMenuImplementation()
                        << "\nretrieved from Event Setup" << std::endl;

            }

        } else {
            // L1 trigger configuration from event setup only
            LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                    << nameAlgoTechTrig
                    << "\not found in the trigger menu \n  "
                    << m_l1GtMenu->gtTriggerMenuImplementation()
                    << "\nretrieved from Event Setup" << std::endl;

        }

        return iError;

    }

    // check here if a positive bit number was retrieved
    // exit in case of negative bit number, before retrieving L1 GT products, saving time

    if (bitNumber < 0) {

        iError = l1ConfCode + 2;

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {
                LogDebug("L1GtUtils") << "\nNegative bit number for "
                        << triggerCategory(trigCategory) << "\n  "
                        << nameAlgoTechTrig << "\nfrom menu \n  "
                        << m_l1GtMenuLite->gtTriggerMenuImplementation()
                        << "\nretrieved from L1GtTriggerMenuLite" << std::endl;

            } else {
                // fall through: L1 trigger configuration from event setup
                LogDebug("L1GtUtils") << "\nNegative bit number for "
                        << triggerCategory(trigCategory) << "\n  "
                        << nameAlgoTechTrig << "\nfrom menu \n  "
                        << m_l1GtMenu->gtTriggerMenuImplementation()
                        << "\nretrieved from Event Setup" << std::endl;

            }

        } else {
            // L1 trigger configuration from event setup only
            LogDebug("L1GtUtils") << "\nNegative bit number for "
                    << triggerCategory(trigCategory) << "\n  "
                    << nameAlgoTechTrig << "\nfrom menu \n  "
                    << m_l1GtMenu->gtTriggerMenuImplementation()
                    << "\nretrieved from Event Setup" << std::endl;

        }

        return iError;
    }


    // retrieve L1GlobalTriggerRecord and 1GlobalTriggerReadoutRecord product
    // intermediate error code for the records
    // the module returns an error code only if both the lite and the readout record are missing

    int iErrorRecord = 0;

    bool validRecord = false;
    bool gtReadoutRecordValid = false;

    edm::Handle<L1GlobalTriggerRecord> gtRecord;
    iEvent.getByLabel(l1GtRecordInputTag, gtRecord);

    if (gtRecord.isValid()) {

        validRecord = true;

    } else {

        iErrorRecord = 10;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nnot found in the event."
                << std::endl;
    }

    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(l1GtReadoutRecordInputTag, gtReadoutRecord);

    if (gtReadoutRecord.isValid()) {

        gtReadoutRecordValid = true;
        validRecord = true;

    } else {

        iErrorRecord = iErrorRecord + 100;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nnot found in the event."
                << std::endl;

    }

    // get the prescale factor index from
    //  L1GlobalTriggerReadoutRecord if valid
    //  if not, from L1GlobalTriggerRecord if valid
    //  else return an error


    int pfIndexTechTrig = -1;
    int pfIndexAlgoTrig = -1;

    if (validRecord) {
        if (gtReadoutRecordValid) {

            pfIndexTechTrig
                    = (gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexTech();
            pfIndexAlgoTrig
                    = (gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo();

        } else {

            pfIndexTechTrig
                    = static_cast<int> (gtRecord->gtPrescaleFactorIndexTech());
            pfIndexAlgoTrig
                    = static_cast<int> (gtRecord->gtPrescaleFactorIndexAlgo());

        }

    } else {

        LogDebug("L1GtUtils") << "\nError: "
                << "\nNo valid L1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nfound in the event."
                << "\nNo valid L1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nfound in the event."
                << std::endl;

        iError = l1ConfCode + iErrorRecord;
        return iError;

    }

    // depending on trigger category (algorithm trigger or technical trigger)
    // get the correct quantities

    // number of sets of prescale factors
    // index of prescale factor set retrieved from data
    // pointer to the actual prescale factor set
    // pointer to the set of trigger masks

    size_t pfSetsSize = 0;
    int pfIndex = -1;
    const std::vector<int>* prescaleFactorsSubset = 0;
    const std::vector<unsigned int>* triggerMasksSet = 0;

    switch (trigCategory) {
        case AlgorithmTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    pfSetsSize = m_prescaleFactorsAlgoTrigLite->size();
                    triggerMasksSet = m_triggerMaskAlgoTrigLite;

                } else {
                    // fall through: L1 trigger configuration from event setup
                    pfSetsSize = m_prescaleFactorsAlgoTrig->size();
                    triggerMasksSet = m_triggerMaskAlgoTrig;

                }

            } else {
                // L1 trigger configuration from event setup only
                pfSetsSize = m_prescaleFactorsAlgoTrig->size();
                triggerMasksSet = m_triggerMaskAlgoTrig;

            }

            pfIndex = pfIndexAlgoTrig;

        }
            break;
        case TechnicalTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    pfSetsSize = m_prescaleFactorsTechTrigLite->size();
                    triggerMasksSet = m_triggerMaskTechTrigLite;

                } else {
                    // fall through: L1 trigger configuration from event setup
                    pfSetsSize = m_prescaleFactorsTechTrig->size();
                    triggerMasksSet = m_triggerMaskTechTrig;

                }

            } else {
                // L1 trigger configuration from event setup only
                pfSetsSize = m_prescaleFactorsTechTrig->size();
                triggerMasksSet = m_triggerMaskTechTrig;

            }

            pfIndex = pfIndexTechTrig;

        }
            break;
        default: {
            // should not be the case
            iError = l1ConfCode + iErrorRecord + 3;
            return iError;

        }
            break;
    }


    // test prescale factor set index correctness, then retrieve the actual set of prescale factors

    if (pfIndex < 0) {

        iError = l1ConfCode + iErrorRecord + 1000;
        LogDebug("L1GtUtils")
                << "\nError: index of prescale factor set retrieved from the data \n"
                << "less than zero."
                << "\n  Value of index retrieved from data = " << pfIndex
                << std::endl;

        return iError;

    } else if (pfIndex >= (static_cast<int>(pfSetsSize))) {
        iError = l1ConfCode + iErrorRecord + 2000;
        LogDebug("L1GtUtils")
                << "\nError: index of prescale factor set retrieved from the data \n"
                << "greater than the size of the vector of prescale factor sets."
                << "\n  Value of index retrieved from data = " << pfIndex
                << "\n  Vector size = " << pfSetsSize << std::endl;

        return iError;

    } else {
        switch (trigCategory) {
            case AlgorithmTrigger: {
                if (m_retrieveL1GtTriggerMenuLite) {
                    if (m_l1GtMenuLiteValid) {
                        prescaleFactorsSubset
                                = &((*m_prescaleFactorsAlgoTrigLite).at(pfIndex));

                    } else {
                        // fall through: L1 trigger configuration from event setup
                        prescaleFactorsSubset
                                = &((*m_prescaleFactorsAlgoTrig).at(pfIndex));

                    }

                } else {
                    // L1 trigger configuration from event setup only
                    prescaleFactorsSubset
                            = &((*m_prescaleFactorsAlgoTrig).at(pfIndex));

                }

            }
                break;
            case TechnicalTrigger: {
                if (m_retrieveL1GtTriggerMenuLite) {
                    if (m_l1GtMenuLiteValid) {
                        prescaleFactorsSubset
                                = &((*m_prescaleFactorsTechTrigLite).at(pfIndex));

                    } else {
                        // fall through: L1 trigger configuration from event setup
                        prescaleFactorsSubset
                                = &((*m_prescaleFactorsTechTrig).at(pfIndex));

                    }

                } else {
                    // L1 trigger configuration from event setup only
                    prescaleFactorsSubset
                            = &((*m_prescaleFactorsTechTrig).at(pfIndex));

                }

            }
                break;
            default: {
                // do nothing - it was tested before, with return

            }
                break;
        }

    }


    // algorithm result before applying the trigger masks
    // the bit number is positive (tested previously)

    switch (trigCategory) {
        case AlgorithmTrigger: {
            if (gtReadoutRecordValid) {
                const DecisionWord& decWord = gtReadoutRecord->decisionWord();
                decisionBeforeMask = trigResult(decWord, bitNumber,
                        nameAlgoTechTrig, trigCategory, iError);
                if (iError) {
                    return (iError + l1ConfCode + iErrorRecord);
                }

            } else {

                const DecisionWord& decWord =
                        gtRecord->decisionWordBeforeMask();
                decisionBeforeMask = trigResult(decWord, bitNumber,
                        nameAlgoTechTrig, trigCategory, iError);
                if (iError) {
                    return (iError + l1ConfCode + iErrorRecord);
                }

            }

        }
            break;
        case TechnicalTrigger: {
            if (gtReadoutRecordValid) {
                const DecisionWord& decWord =
                        gtReadoutRecord->technicalTriggerWord();
                decisionBeforeMask = trigResult(decWord, bitNumber,
                        nameAlgoTechTrig, trigCategory, iError);
                if (iError) {
                    return (iError + l1ConfCode + iErrorRecord);
                }

            } else {

                const DecisionWord& decWord =
                        gtRecord->technicalTriggerWordBeforeMask();
                decisionBeforeMask = trigResult(decWord, bitNumber,
                        nameAlgoTechTrig, trigCategory, iError);
                if (iError) {
                    return (iError + l1ConfCode + iErrorRecord);
                }

            }

        }
            break;
        default: {
            // do nothing - it was tested before, with return

        }
            break;
    }

    // prescale factor
    // the bit number is positive (tested previously)

    if (bitNumber < (static_cast<int> (prescaleFactorsSubset->size()))) {
        prescaleFactor = (*prescaleFactorsSubset)[bitNumber];
    } else {
        iError = l1ConfCode + iErrorRecord + 4000;
        LogDebug("L1GtUtils") << "\nError: bit number " << bitNumber
                << " retrieved for " << triggerCategory(trigCategory) << "\n  "
                << nameAlgoTechTrig
                << "\ngreater than size of actual L1 GT prescale factor set: "
                << prescaleFactorsSubset->size()
                << "\nError: Inconsistent L1 trigger configuration!"
                << std::endl;

        return iError;
    }

    // trigger mask and trigger result after applying the trigger masks

    if (bitNumber < (static_cast<int> ((*triggerMasksSet).size()))) {

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {
                triggerMask = (*triggerMasksSet)[bitNumber];

            } else {
                // fall through: L1 trigger configuration from event setup
                // masks in event setup are for all partitions
                triggerMask = ((*triggerMasksSet)[bitNumber]) & (1
                        << m_physicsDaqPartition);

            }

        } else {
            // L1 trigger configuration from event setup only
            // masks in event setup are for all partitions
            triggerMask = ((*triggerMasksSet)[bitNumber]) & (1
                    << m_physicsDaqPartition);

        }


    } else {
        iError = l1ConfCode + iErrorRecord + 5000;
        LogDebug("L1GtUtils") << "\nError: bit number " << bitNumber
                << " retrieved for " << triggerCategory(trigCategory) << "\n  "
                << nameAlgoTechTrig
                << "\ngreater than size of L1 GT trigger mask set: "
                << (*triggerMasksSet).size()
                << "\nError: Inconsistent L1 trigger configuration!"
                << std::endl;

        return iError;

    }

    decisionAfterMask = decisionBeforeMask;

    if (triggerMask) {
        decisionAfterMask = false;
    }

    return iError;

}


const int L1GtUtils::l1Results(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, bool& decisionBeforeMask,
        bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const {

    // initial values for returned results
    decisionBeforeMask = false;
    decisionAfterMask = false;
    prescaleFactor = -1;
    triggerMask = -1;

    int l1ErrorCode = 0;

    l1ErrorCode = l1Results(iEvent, m_provL1GtRecordInputTag,
            m_provL1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return l1ErrorCode;

}

//

const bool L1GtUtils::decisionBeforeMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionBeforeMask;

}

const bool L1GtUtils::decisionBeforeMask(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionBeforeMask;

}

//

const bool L1GtUtils::decisionAfterMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

const bool L1GtUtils::decisionAfterMask(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

//

const bool L1GtUtils::decision(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

const bool L1GtUtils::decision(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return decisionAfterMask;

}

//

const int L1GtUtils::prescaleFactor(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return prescaleFactor;

}

const int L1GtUtils::prescaleFactor(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return prescaleFactor;

}

const int L1GtUtils::triggerMask(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return triggerMask;

}

const int L1GtUtils::triggerMask(const edm::Event& iEvent,
        const std::string& nameAlgoTechTrig, int& errorCode) const {

    // initial values
    bool decisionBeforeMask = false;
    bool decisionAfterMask = false;
    int prescaleFactor = -1;
    int triggerMask = -1;

    errorCode = l1Results(iEvent, nameAlgoTechTrig, decisionBeforeMask,
            decisionAfterMask, prescaleFactor, triggerMask);

    return triggerMask;

}

const int L1GtUtils::triggerMask(const std::string& nameAlgoTechTrig,
        int& errorCode) const {

    // initial values for returned results
    int triggerMaskValue = -1;

    // initialize error code and L1 configuration code
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        errorCode = iError;
        return triggerMaskValue;
    }

    // at this point, a valid L1 configuration is available, so the if/else if/else
    // can be simplified

    // if the given name is not an algorithm trigger alias, an algorithm trigger name
    // or a technical trigger in the current menu, return with error code 1

    TriggerCategory trigCategory = AlgorithmTrigger;
    int bitNumber = -1;

    if (!l1AlgoTechTrigBitNumber(nameAlgoTechTrig, trigCategory, bitNumber)) {

        iError = l1ConfCode + 1;

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {

                LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                        << nameAlgoTechTrig
                        << "\not found in the trigger menu \n  "
                        << m_l1GtMenuLite->gtTriggerMenuImplementation()
                        << "\nretrieved from L1GtTriggerMenuLite" << std::endl;

            } else {

                // fall through: L1 trigger configuration from event setup
                LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                        << nameAlgoTechTrig
                        << "\not found in the trigger menu \n  "
                        << m_l1GtMenu->gtTriggerMenuImplementation()
                        << "\nretrieved from Event Setup" << std::endl;

            }

        } else {
            // L1 trigger configuration from event setup only
            LogDebug("L1GtUtils") << "\nAlgorithm/technical trigger \n  "
                    << nameAlgoTechTrig
                    << "\not found in the trigger menu \n  "
                    << m_l1GtMenu->gtTriggerMenuImplementation()
                    << "\nretrieved from Event Setup" << std::endl;

        }

        errorCode = iError;
        return triggerMaskValue;

    }

    // check here if a positive bit number was retrieved
    // exit in case of negative bit number, before retrieving L1 GT products, saving time

    if (bitNumber < 0) {

        iError = l1ConfCode + 2;

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {
                LogDebug("L1GtUtils") << "\nNegative bit number for "
                        << triggerCategory(trigCategory) << "\n  "
                        << nameAlgoTechTrig << "\nfrom menu \n  "
                        << m_l1GtMenuLite->gtTriggerMenuImplementation()
                        << "\nretrieved from L1GtTriggerMenuLite" << std::endl;

            } else {
                // fall through: L1 trigger configuration from event setup
                LogDebug("L1GtUtils") << "\nNegative bit number for "
                        << triggerCategory(trigCategory) << "\n  "
                        << nameAlgoTechTrig << "\nfrom menu \n  "
                        << m_l1GtMenu->gtTriggerMenuImplementation()
                        << "\nretrieved from Event Setup" << std::endl;

            }

        } else {
            // L1 trigger configuration from event setup only
            LogDebug("L1GtUtils") << "\nNegative bit number for "
                    << triggerCategory(trigCategory) << "\n  "
                    << nameAlgoTechTrig << "\nfrom menu \n  "
                    << m_l1GtMenu->gtTriggerMenuImplementation()
                    << "\nretrieved from Event Setup" << std::endl;

        }

        errorCode = iError;
        return triggerMaskValue;
    }

    // depending on trigger category (algorithm trigger or technical trigger)
    // get the correct quantities

    // pointer to the set of trigger masks

    const std::vector<unsigned int>* triggerMasksSet = 0;

    switch (trigCategory) {
        case AlgorithmTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    triggerMasksSet = m_triggerMaskAlgoTrigLite;

                } else {
                    // fall through: L1 trigger configuration from event setup
                    triggerMasksSet = m_triggerMaskAlgoTrig;

                }

            } else {
                // L1 trigger configuration from event setup only
                triggerMasksSet = m_triggerMaskAlgoTrig;

            }

        }
            break;
        case TechnicalTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    triggerMasksSet = m_triggerMaskTechTrigLite;

                } else {
                    // fall through: L1 trigger configuration from event setup
                    triggerMasksSet = m_triggerMaskTechTrig;

                }

            } else {
                // L1 trigger configuration from event setup only
                triggerMasksSet = m_triggerMaskTechTrig;

            }

        }
            break;
        default: {
            // should not be the case
            iError = l1ConfCode + 3;

            errorCode = iError;
            return triggerMaskValue;

        }
            break;
    }

    // trigger mask

    if (bitNumber < (static_cast<int> ((*triggerMasksSet).size()))) {

        if (m_retrieveL1GtTriggerMenuLite) {
            if (m_l1GtMenuLiteValid) {
                triggerMaskValue = (*triggerMasksSet)[bitNumber];

            } else {
                // fall through: L1 trigger configuration from event setup
                // masks in event setup are for all partitions
                triggerMaskValue = ((*triggerMasksSet)[bitNumber]) & (1
                        << m_physicsDaqPartition);

            }

        } else {
            // L1 trigger configuration from event setup only
            // masks in event setup are for all partitions
            triggerMaskValue = ((*triggerMasksSet)[bitNumber]) & (1
                    << m_physicsDaqPartition);

        }

    } else {
        iError = l1ConfCode + 5000;
        LogDebug("L1GtUtils") << "\nError: bit number " << bitNumber
                << " retrieved for " << triggerCategory(trigCategory) << "\n  "
                << nameAlgoTechTrig
                << "\ngreater than size of L1 GT trigger mask set: "
                << (*triggerMasksSet).size()
                << "\nError: Inconsistent L1 trigger configuration!"
                << std::endl;

        errorCode = iError;
        return triggerMaskValue;

    }

    errorCode = iError;
    return triggerMaskValue;

}

const int L1GtUtils::prescaleFactorSetIndex(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const TriggerCategory& trigCategory, int& errorCode) const {

    // initialize the index to a negative value
    int pfIndex = -1;

    // initialize error code and L1 configuration code
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        errorCode = iError;
        return pfIndex;
    }

    // at this point, a valid L1 configuration is available, so the if/else if/else
    // can be simplified

    // retrieve L1GlobalTriggerRecord and 1GlobalTriggerReadoutRecord product
    // intermediate error code for the records
    // the module returns an error code only if both the lite and the readout record are missing

    int iErrorRecord = 0;

    bool validRecord = false;
    bool gtReadoutRecordValid = false;

    edm::Handle<L1GlobalTriggerRecord> gtRecord;
    iEvent.getByLabel(l1GtRecordInputTag, gtRecord);

    if (gtRecord.isValid()) {

        validRecord = true;

    } else {

        iErrorRecord = 10;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nnot found in the event."
                << std::endl;
    }

    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(l1GtReadoutRecordInputTag, gtReadoutRecord);

    if (gtReadoutRecord.isValid()) {

        gtReadoutRecordValid = true;
        validRecord = true;

    } else {

        iErrorRecord = iErrorRecord + 100;
        LogDebug("L1GtUtils") << "\nL1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nnot found in the event."
                << std::endl;

    }

    // get the prescale factor index from
    //  L1GlobalTriggerReadoutRecord if valid
    //  if not, from L1GlobalTriggerRecord if valid
    //  else return an error


    int pfIndexTechTrig = -1;
    int pfIndexAlgoTrig = -1;

    if (validRecord) {
        if (gtReadoutRecordValid) {

            pfIndexTechTrig
                    = (gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexTech();
            pfIndexAlgoTrig
                    = (gtReadoutRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo();

        } else {

            pfIndexTechTrig
                    = static_cast<int> (gtRecord->gtPrescaleFactorIndexTech());
            pfIndexAlgoTrig
                    = static_cast<int> (gtRecord->gtPrescaleFactorIndexAlgo());

        }

    } else {

        LogDebug("L1GtUtils") << "\nError: "
                << "\nNo valid L1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nfound in the event."
                << "\nNo valid L1GlobalTriggerReadoutRecord with \n  "
                << l1GtReadoutRecordInputTag << "\nfound in the event."
                << std::endl;

        iError = l1ConfCode + iErrorRecord;

        errorCode = iError;
        return pfIndex;

    }

    // depending on trigger category (algorithm trigger or technical trigger)
    // get the correct quantities

    // number of sets of prescale factors
    // index of prescale factor set retrieved from data
    // pointer to the actual prescale factor set
    // pointer to the set of trigger masks

    size_t pfSetsSize = 0;

    switch (trigCategory) {
        case AlgorithmTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    pfSetsSize = m_prescaleFactorsAlgoTrigLite->size();

                } else {
                    // fall through: L1 trigger configuration from event setup
                    pfSetsSize = m_prescaleFactorsAlgoTrig->size();

                }

            } else {
                // L1 trigger configuration from event setup only
                pfSetsSize = m_prescaleFactorsAlgoTrig->size();

            }

            pfIndex = pfIndexAlgoTrig;

        }
            break;
        case TechnicalTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    pfSetsSize = m_prescaleFactorsTechTrigLite->size();

                } else {
                    // fall through: L1 trigger configuration from event setup
                    pfSetsSize = m_prescaleFactorsTechTrig->size();

                }

            } else {
                // L1 trigger configuration from event setup only
                pfSetsSize = m_prescaleFactorsTechTrig->size();

            }

            pfIndex = pfIndexTechTrig;

        }
            break;
        default: {
            // should not be the case
            iError = l1ConfCode + iErrorRecord + 3;
            return iError;

        }
            break;
    }


    // test prescale factor set index correctness, then retrieve the actual set of prescale factors

    if (pfIndex < 0) {

        iError = l1ConfCode + iErrorRecord + 1000;
        LogDebug("L1GtUtils")
                << "\nError: index of prescale factor set retrieved from the data \n"
                << "less than zero."
                << "\n  Value of index retrieved from data = " << pfIndex
                << std::endl;

        errorCode = iError;
        return pfIndex;

    } else if (pfIndex >= (static_cast<int>(pfSetsSize))) {
        iError = l1ConfCode + iErrorRecord + 2000;
        LogDebug("L1GtUtils")
                << "\nError: index of prescale factor set retrieved from the data \n"
                << "greater than the size of the vector of prescale factor sets."
                << "\n  Value of index retrieved from data = " << pfIndex
                << "\n  Vector size = " << pfSetsSize << std::endl;

        errorCode = iError;
        return pfIndex;

    } else {

        errorCode = iError;
        return pfIndex;
    }

    errorCode = iError;
    return pfIndex;

}


const int L1GtUtils::prescaleFactorSetIndex(const edm::Event& iEvent,
        const TriggerCategory& trigCategory, int& errorCode) const {

    // initialize error code and return value
    int iError = 0;
    int pfIndex = -1;

    pfIndex = prescaleFactorSetIndex(iEvent, m_provL1GtRecordInputTag,
            m_provL1GtReadoutRecordInputTag, trigCategory, iError);

    // return the error code and the index value
    // if the  error code is 0, the index returned is -1
    errorCode = iError;
    return pfIndex;

}



const std::vector<int>& L1GtUtils::prescaleFactorSet(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag,
        const TriggerCategory& trigCategory, int& errorCode) {

    // clear the vector before filling it
    m_prescaleFactorSet.clear();

    // initialize error code
    int iError = 0;

    const int pfIndex = prescaleFactorSetIndex(iEvent, l1GtRecordInputTag,
            l1GtReadoutRecordInputTag, trigCategory, iError);

    if (iError == 0) {

        switch (trigCategory) {
            case AlgorithmTrigger: {
                if (m_retrieveL1GtTriggerMenuLite) {
                    if (m_l1GtMenuLiteValid) {
                        m_prescaleFactorSet
                                = (*m_prescaleFactorsAlgoTrigLite).at(pfIndex);

                    } else {
                        // fall through: L1 trigger configuration from event setup
                        m_prescaleFactorSet = (*m_prescaleFactorsAlgoTrig).at(
                                pfIndex);

                    }

                } else {
                    // L1 trigger configuration from event setup only
                    m_prescaleFactorSet = (*m_prescaleFactorsAlgoTrig).at(
                            pfIndex);

                }

            }
                break;
            case TechnicalTrigger: {
                if (m_retrieveL1GtTriggerMenuLite) {
                    if (m_l1GtMenuLiteValid) {
                        m_prescaleFactorSet
                                = (*m_prescaleFactorsTechTrigLite).at(pfIndex);

                    } else {
                        // fall through: L1 trigger configuration from event setup
                        m_prescaleFactorSet = (*m_prescaleFactorsTechTrig).at(
                                pfIndex);

                    }

                } else {
                    // L1 trigger configuration from event setup only
                    m_prescaleFactorSet = (*m_prescaleFactorsTechTrig).at(
                            pfIndex);

                }

            }
                break;
            default: {
                // do nothing - it was tested before, with return

            }
                break;
        }

    }

    errorCode = iError;
    return m_prescaleFactorSet;

}

const std::vector<int>& L1GtUtils::prescaleFactorSet(const edm::Event& iEvent,
        const TriggerCategory& trigCategory, int& errorCode) {

    // clear the vector before filling it
    m_prescaleFactorSet.clear();

    // initialize error code
    int iError = 0;

    m_prescaleFactorSet = prescaleFactorSet(iEvent, m_provL1GtRecordInputTag,
            m_provL1GtReadoutRecordInputTag, trigCategory, iError);

    errorCode = iError;
    return m_prescaleFactorSet;

}




const std::vector<unsigned int>& L1GtUtils::triggerMaskSet(
        const TriggerCategory& trigCategory, int& errorCode) {

    // clear the vector before filling it
    m_triggerMaskSet.clear();

    // initialize error code and L1 configuration code
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        errorCode = iError;
        return m_triggerMaskSet;
    }

    // at this point, a valid L1 configuration is available, so the if/else if/else
    // can be simplified


    // depending on trigger category (algorithm trigger or technical trigger)
    // get the correct quantities

    // pointer to the set of trigger masks

    switch (trigCategory) {
        case AlgorithmTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                // L1GtTriggerMenuLite has masks for physics partition only
                // avoid copy to m_triggerMaskSet, return directly m_triggerMaskAlgoTrigLite
               if (m_l1GtMenuLiteValid) {
                    errorCode = iError;
                    return (*m_triggerMaskAlgoTrigLite);

                } else {
                    // fall through: L1 trigger configuration from event setup
                    for (unsigned i = 0; i < m_triggerMaskAlgoTrig->size(); i++) {
                        m_triggerMaskSet.push_back(
                                ((*m_triggerMaskAlgoTrig)[i]) & (1
                                        << m_physicsDaqPartition));
                    }

                }

            } else {
                // L1 trigger configuration from event setup only
                for (unsigned i = 0; i < m_triggerMaskAlgoTrig->size(); i++) {
                    m_triggerMaskSet.push_back(((*m_triggerMaskAlgoTrig)[i])
                            & (1 << m_physicsDaqPartition));
                }

            }
        }
            break;
        case TechnicalTrigger: {
            if (m_retrieveL1GtTriggerMenuLite) {
                if (m_l1GtMenuLiteValid) {
                    errorCode = iError;
                    return (*m_triggerMaskTechTrigLite);

                } else {
                    // fall through: L1 trigger configuration from event setup
                    for (unsigned i = 0; i < m_triggerMaskTechTrig->size(); i++) {
                        m_triggerMaskSet.push_back(
                                ((*m_triggerMaskTechTrig)[i]) & (1
                                        << m_physicsDaqPartition));
                    }

                }

            } else {
                // L1 trigger configuration from event setup only
                for (unsigned i = 0; i < m_triggerMaskTechTrig->size(); i++) {
                    m_triggerMaskSet.push_back(((*m_triggerMaskTechTrig)[i])
                            & (1 << m_physicsDaqPartition));
                }

            }
        }
            break;
        default: {
            // should not be the case
            iError = l1ConfCode + 3;

            errorCode = iError;
            return m_triggerMaskSet;

        }
            break;
    }

    errorCode = iError;
    return m_triggerMaskSet;

}



const std::string& L1GtUtils::l1TriggerMenu() const {

    if (m_retrieveL1GtTriggerMenuLite) {
        if (m_l1GtMenuLiteValid) {
            return m_l1GtMenuLite->gtTriggerMenuName();

        } else if (m_retrieveL1EventSetup) {
            return m_l1GtMenu->gtTriggerMenuName();

        } else {
            // only L1GtTriggerMenuLite requested, but it is not valid
            return EmptyString;

        }
    } else if (m_retrieveL1EventSetup) {
        return m_l1GtMenu->gtTriggerMenuName();

    } else {
        // L1 trigger configuration not retrieved
        return EmptyString;

    }

}

const std::string& L1GtUtils::l1TriggerMenuImplementation() const {

    if (m_retrieveL1GtTriggerMenuLite) {
        if (m_l1GtMenuLiteValid) {
            return m_l1GtMenuLite->gtTriggerMenuImplementation();

        } else if (m_retrieveL1EventSetup) {
            return m_l1GtMenu->gtTriggerMenuImplementation();

        } else {
            // only L1GtTriggerMenuLite requested, but it is not valid
            return EmptyString;

        }
    } else if (m_retrieveL1EventSetup) {
        return m_l1GtMenu->gtTriggerMenuImplementation();

    } else {
        // L1 trigger configuration not retrieved
        return EmptyString;

    }

}

const L1GtTriggerMenu* L1GtUtils::ptrL1TriggerMenuEventSetup(int& errorCode) {

    // initialize error code and return value
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        errorCode = iError;
        return 0;
    }

    if (m_retrieveL1EventSetup) {
        errorCode = iError;
        return m_l1GtMenu;
    } else {
        iError = l1ConfCode;

        errorCode = iError;
        return 0;

    }

    errorCode = iError;
    return m_l1GtMenu;
}

const L1GtTriggerMenuLite* L1GtUtils::ptrL1GtTriggerMenuLite(int& errorCode) {

    // initialize error code and return value
    int iError = 0;
    int l1ConfCode = 0;

    // check if L1 configuration is available

    if (!availableL1Configuration(iError, l1ConfCode)) {
        errorCode = iError;
        return 0;
    }

    if (m_retrieveL1GtTriggerMenuLite) {
        if (m_l1GtMenuLiteValid) {

            errorCode = iError;
            return m_l1GtMenuLite;

        } else {
            iError = l1ConfCode;

            errorCode = iError;
            return 0;
        }
    } else {
        iError = l1ConfCode;

        errorCode = iError;
        return 0;
    }

    errorCode = iError;
    return m_l1GtMenuLite;

}

const bool L1GtUtils::availableL1Configuration(int& errorCode, int& l1ConfCode) const {

    if (m_retrieveL1GtTriggerMenuLite) {
        if (!m_retrieveL1EventSetup) {
            LogDebug("L1GtUtils")
                    << "\nRetrieve L1 trigger configuration from L1GtTriggerMenuLite only.\n"
                    << std::endl;
            l1ConfCode = 0;
        } else {
            LogDebug("L1GtUtils")
                    << "\nFall through: retrieve L1 trigger configuration from L1GtTriggerMenuLite."
                    << "\nIf L1GtTriggerMenuLite not valid, try to retrieve from event setup.\n"
                    << std::endl;
            l1ConfCode = 100000;
        }

        if (m_l1GtMenuLiteValid) {
            LogDebug("L1GtUtils")
                    << "\nRetrieve L1 trigger configuration from L1GtTriggerMenuLite, valid product.\n"
                    << std::endl;
            l1ConfCode = l1ConfCode  + 10000;
            errorCode = 0;

            return true;

        } else if (m_retrieveL1EventSetup) {
            if (m_l1EventSetupValid) {
                LogDebug("L1GtUtils")
                        << "\nFall through: retrieve L1 trigger configuration from event setup."
                        << "\nFirst option was L1GtTriggerMenuLite - but product is not valid.\n"
                        << std::endl;
                l1ConfCode = l1ConfCode  + 20000;
                errorCode = 0;

                return true;

            } else {
                LogDebug("L1GtUtils")
                        << "\nFall through: L1GtTriggerMenuLite not valid, event setup not valid.\n"
                        << std::endl;
                l1ConfCode = l1ConfCode  + L1GtNotValidError;
                errorCode = l1ConfCode;

                return false;


            }

        } else {
            LogDebug("L1GtUtils")
                    << "\nError: L1 trigger configuration requested from L1GtTriggerMenuLite only"
                    << "\nbut L1GtTriggerMenuLite is not valid.\n" << std::endl;
            l1ConfCode = l1ConfCode  + L1GtNotValidError;
            errorCode = l1ConfCode;

            return false;

        }
    } else if (m_retrieveL1EventSetup) {

        LogDebug("L1GtUtils")
                << "\nRetrieve L1 trigger configuration from event setup."
                << "\nL1GtTriggerMenuLite product was not requested.\n"
                << std::endl;
        l1ConfCode = 200000;

        if (m_l1EventSetupValid) {
            LogDebug("L1GtUtils")
                    << "\nRetrieve L1 trigger configuration from event setup only."
                    << "\nValid L1 trigger event setup.\n"
                    << std::endl;
            l1ConfCode = l1ConfCode  + 10000;
            errorCode = 0;

            return true;

        } else {
            LogDebug("L1GtUtils")
                    << "\nRetrieve L1 trigger configuration from event setup only."
                    << "\nNo valid L1 trigger event setup.\n"
                    << std::endl;
            l1ConfCode = l1ConfCode  + L1GtNotValidError;
            errorCode = l1ConfCode;

            return false;


        }

    } else {
        LogDebug("L1GtUtils")
                << "\nError: no L1 trigger configuration requested to be retrieved."
                << "\nMust call before getL1GtRunCache in beginRun and analyze.\n"
                << std::endl;
        l1ConfCode = 300000;
        errorCode = l1ConfCode;

        return false;

    }
}

// private methods

const bool L1GtUtils::trigResult(const DecisionWord& decWord,
        const int bitNumber, const std::string& nameAlgoTechTrig,
        const TriggerCategory& trigCategory, int& errorCode) const {

    bool trigRes = false;
    errorCode = 0;

    if (bitNumber < (static_cast<int> (decWord.size()))) {
        trigRes = decWord[bitNumber];
    } else {
        errorCode = 3000;
        LogDebug("L1GtUtils") << "\nError: bit number " << bitNumber
                << " retrieved for " << triggerCategory(trigCategory) << "\n  "
                << nameAlgoTechTrig
                << "\ngreater than size of L1 GT decision word: "
                << decWord.size()
                << "\nError: Inconsistent L1 trigger configuration!"
                << std::endl;
    }

    return trigRes;
}

L1GtUtils::LogicalExpressionL1Results::LogicalExpressionL1Results(
        const std::string& expression, L1GtUtils& l1GtUtils) :

        m_logicalExpression(expression),

        m_l1GtUtils(l1GtUtils),

        m_l1GtRecordInputTag(edm::InputTag()),

        m_l1GtReadoutRecordInputTag(edm::InputTag()),

        m_l1ConfCode(-1),

        m_validL1Configuration(false),

        m_validLogicalExpression(false),

        m_l1GtInputTagsFromProv(true),

        m_l1ResultsAlreadyCalled(false),

        m_expL1TriggersSize(0),

        m_expBitsTechTrigger(false) {

    initialize();
}

L1GtUtils::LogicalExpressionL1Results::LogicalExpressionL1Results(
        const std::string& expression, L1GtUtils& l1GtUtils,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag) :

        m_logicalExpression(expression),

        m_l1GtUtils(l1GtUtils),

        m_l1GtRecordInputTag(l1GtRecordInputTag),

        m_l1GtReadoutRecordInputTag(l1GtReadoutRecordInputTag),

        m_l1ConfCode(-1),

        m_validL1Configuration(false),

        m_validLogicalExpression(false),

        m_l1GtInputTagsFromProv(false),

        m_l1ResultsAlreadyCalled(false),

        m_expL1TriggersSize(0),

        m_expBitsTechTrigger(false) {

    initialize();
}

// destructor
L1GtUtils::LogicalExpressionL1Results::~LogicalExpressionL1Results() {

    // empty

}

bool L1GtUtils::LogicalExpressionL1Results::initialize() {

    // get the vector of triggers corresponding to the logical expression
    // check also the logical expression - add/remove spaces if needed

    try {

        L1GtLogicParser m_l1AlgoLogicParser = L1GtLogicParser(
                m_logicalExpression);

        // list of L1 triggers from the logical expression
        m_expL1Triggers = m_l1AlgoLogicParser.operandTokenVector();
        m_expL1TriggersSize = m_expL1Triggers.size();

        m_validLogicalExpression = true;

    } catch (cms::Exception & ex) {
        m_validLogicalExpression = false;

        LogDebug("L1GtUtils") << ex;
        LogDebug("L1GtUtils") << ex.what();
        LogDebug("L1GtUtils") << ex.explainSelf();
    }

    // try to convert the string representing each L1 trigger to bit number,
    //   to check if the logical expression is constructed from bit numbers
    // trade-off: cache it here, irrespective of the expression
    //   when the conversion fails (normally for the first seed, 
    //   if not expression of technical trigger bits), stop and 
    //   set m_expBitsTechTrigger to false

    m_expBitsTechTrigger = true;

    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {
        const std::string& bitString = (m_expL1Triggers[iTrig]).tokenName;
        std::istringstream bitStream(bitString);
        int bitInt;

        if ((bitStream >> bitInt).fail()) {

            m_expBitsTechTrigger = false;

            break;
        }

        (m_expL1Triggers[iTrig]).tokenNumber = bitInt;

    }

    // resize and fill 
    m_decisionsBeforeMask.resize(m_expL1TriggersSize);
    m_decisionsAfterMask.resize(m_expL1TriggersSize);
    m_prescaleFactors.resize(m_expL1TriggersSize);
    m_triggerMasks.resize(m_expL1TriggersSize);
    m_errorCodes.resize(m_expL1TriggersSize);
    m_expTriggerCategory.resize(m_expL1TriggersSize);
    m_expTriggerInMenu.resize(m_expL1TriggersSize);

    LogDebug("L1GtUtils") << std::endl;
    LogTrace("L1GtUtils") << "\nLogical expression\n  " << m_logicalExpression
            << "\n has " << m_expL1TriggersSize << " L1 triggers" << std::endl;
    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {

        const std::string& trigNameOrAlias = (m_expL1Triggers[iTrig]).tokenName;
        LogTrace("L1GtUtils") << "  " << trigNameOrAlias << std::endl;

        (m_decisionsBeforeMask[iTrig]).first = trigNameOrAlias;
        (m_decisionsBeforeMask[iTrig]).second = false;

        (m_decisionsAfterMask[iTrig]).first = trigNameOrAlias;
        (m_decisionsAfterMask[iTrig]).second = false;

        (m_prescaleFactors[iTrig]).first = trigNameOrAlias;
        (m_prescaleFactors[iTrig]).second = -1;

        (m_triggerMasks[iTrig]).first = trigNameOrAlias;
        (m_triggerMasks[iTrig]).second = -1;

        (m_errorCodes[iTrig]).first = trigNameOrAlias;
        (m_errorCodes[iTrig]).second = -1;

        m_expTriggerCategory[iTrig] = L1GtUtils::AlgorithmTrigger;

        m_expTriggerInMenu[iTrig] = false;

    }
    LogTrace("L1GtUtils") << std::endl;

    return true;

}


const int L1GtUtils::LogicalExpressionL1Results::logicalExpressionRunUpdate(
        const edm::Run& iRun, const edm::EventSetup& evSetup,
        const std::string& logicExpression) {

    // initialize error code
    int errorCode = 0;

    // logical expression has changed - one must re-initialize all quantities related to the logical expression
    // and clear the vectors

    m_logicalExpression = logicExpression;
    m_validLogicalExpression = false;

    m_l1ResultsAlreadyCalled = false;

    m_expL1TriggersSize = 0;
    m_expBitsTechTrigger = false;
    
    // 
    m_decisionsBeforeMask.clear();
    m_decisionsAfterMask.clear();
    m_prescaleFactors.clear();
    m_triggerMasks.clear();
    m_errorCodes.clear();
    m_expTriggerCategory.clear();
    m_expTriggerInMenu.clear();


    initialize();

    //
    errorCode = logicalExpressionRunUpdate(iRun, evSetup);

    return errorCode;

}


const int L1GtUtils::LogicalExpressionL1Results::logicalExpressionRunUpdate(
        const edm::Run& iRun, const edm::EventSetup& evSetup) {

    // check first that a valid L1 configuration was retrieved, 
    // to prevent also calls before the L1 configuration retrieval

    // initialize error code and L1 configuration code
    int errorCode = 0;
    int l1ConfCode = 0;

    if (!(m_l1GtUtils.availableL1Configuration(errorCode, l1ConfCode))) {

        m_validL1Configuration = false;
        return errorCode;
    } else {

        m_validL1Configuration = true;
        m_l1ConfCode = l1ConfCode;
    }

    // check if the trigger (name of alias) from the logical expression are in the menu, 
    // if names are used, set tokenNumber to the corresponding bit number
    // if technical trigger bits, set tokenName to the corresponding technical trigger name, if 
    //   a technical trigger exists on that bit
    // for each trigger, set also the trigger category

    // initialization 
    L1GtUtils::TriggerCategory trigCategory = L1GtUtils::AlgorithmTrigger;
    int bitNumber = -1;

    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {

        trigCategory = L1GtUtils::AlgorithmTrigger;
        bitNumber = -1;
        const std::string& trigNameOrAlias = (m_expL1Triggers[iTrig]).tokenName;

        if (!m_expBitsTechTrigger) {
            const bool triggerInMenu = m_l1GtUtils.l1AlgoTechTrigBitNumber(
                    trigNameOrAlias, trigCategory, bitNumber);

            (m_expL1Triggers[iTrig]).tokenNumber = bitNumber;
            m_expTriggerCategory[iTrig] = trigCategory;
            m_expTriggerInMenu[iTrig] = triggerInMenu;

        } else {

            std::string aliasL1Trigger;
            std::string nameL1Trigger;

            trigCategory = L1GtUtils::TechnicalTrigger;
            bitNumber = (m_expL1Triggers[iTrig]).tokenNumber;

            const bool triggerInMenu = m_l1GtUtils.l1TriggerNameFromBit(
                    bitNumber, trigCategory, aliasL1Trigger, nameL1Trigger);

            if (!triggerInMenu) {
                aliasL1Trigger = "Technical_trigger_bit_"
                        + (m_expL1Triggers[iTrig]).tokenName + "_empty";
            }

            (m_expL1Triggers[iTrig]).tokenName = aliasL1Trigger;
            m_expTriggerCategory[iTrig] = trigCategory;
            m_expTriggerInMenu[iTrig] = triggerInMenu;

            // put the names of the technical triggers in the returned quantities

            (m_decisionsBeforeMask[iTrig]).first = aliasL1Trigger;
            (m_decisionsAfterMask[iTrig]).first = aliasL1Trigger;
            (m_prescaleFactors[iTrig]).first = aliasL1Trigger;
            (m_triggerMasks[iTrig]).first = aliasL1Trigger;
            (m_errorCodes[iTrig]).first = aliasL1Trigger;

        }
    }

    return errorCode;

}

const std::vector<std::pair<std::string, bool> >& L1GtUtils::LogicalExpressionL1Results::decisionsBeforeMask() {

    // throw an exception if the result is not computed once per event - user usage error
    if (!m_l1ResultsAlreadyCalled) {
        throw cms::Exception("FailModule") << "\nUsage error: "
                << "\n  Method 'errorCodes' must be called in the event loop before attempting to use this method.\n"
                << std::endl;
    }

    return m_decisionsBeforeMask;

}

const std::vector<std::pair<std::string, bool> >& L1GtUtils::LogicalExpressionL1Results::decisionsAfterMask() {

    // throw an exception if the result is not computed once per event - user usage error
    if (!m_l1ResultsAlreadyCalled) {
        throw cms::Exception("FailModule") << "\nUsage error: "
                << "\n  Method 'errorCodes' must be called in the event loop before attempting to use this method.\n"
                << std::endl;
    }

    return m_decisionsAfterMask;

}

const std::vector<std::pair<std::string, int> >& L1GtUtils::LogicalExpressionL1Results::prescaleFactors() {

    // throw an exception if the result is not computed once per event - user usage error
    if (!m_l1ResultsAlreadyCalled) {
        throw cms::Exception("FailModule") << "\nUsage error: "
                << "\n  Method 'errorCodes' must be called in the event loop before attempting to use this method.\n"
                << std::endl;
    }

    return m_prescaleFactors;

}

const std::vector<std::pair<std::string, int> >& L1GtUtils::LogicalExpressionL1Results::triggerMasks() {

    // throw an exception if the result is not computed once per event - user usage error
    if (!m_l1ResultsAlreadyCalled) {
        throw cms::Exception("FailModule") << "\nUsage error: "
                << "\n  Method 'errorCodes' must be called in the event loop before attempting to use this method.\n"
                << std::endl;
    }

    return m_triggerMasks;

}

const std::vector<std::pair<std::string, int> >& L1GtUtils::LogicalExpressionL1Results::errorCodes(
        const edm::Event& iEvent) {

    m_l1ResultsAlreadyCalled = false;

    // if not a valid L1 configuration, reset all quantities and return
    if (!m_validL1Configuration) {
        reset(m_decisionsBeforeMask);
        reset(m_decisionsAfterMask);
        reset(m_prescaleFactors);
        reset(m_triggerMasks);
        reset(m_errorCodes);

        m_l1ResultsAlreadyCalled = true;
        return m_errorCodes;

    }

    if (m_l1GtInputTagsFromProv) {

        l1Results(iEvent, m_l1GtUtils.provL1GtRecordInputTag(),
                m_l1GtUtils.provL1GtReadoutRecordInputTag());

    } else {

        l1Results(iEvent, m_l1GtRecordInputTag, m_l1GtReadoutRecordInputTag);
    }

    m_l1ResultsAlreadyCalled = true;

    return m_errorCodes;

}

void L1GtUtils::LogicalExpressionL1Results::reset(
        const std::vector<std::pair<std::string, bool> >& _pairVector) const {
    std::vector<std::pair<std::string, bool> > pairVector = _pairVector;
    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {
        (pairVector[iTrig]).second = false;
    }
}

void L1GtUtils::LogicalExpressionL1Results::reset(
        const std::vector<std::pair<std::string, int> >& _pairVector) const {
    std::vector<std::pair<std::string, int> > pairVector = _pairVector;
    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {
        (pairVector[iTrig]).second = -1;
    }
}

void L1GtUtils::LogicalExpressionL1Results::l1Results(const edm::Event& iEvent,
        const edm::InputTag& l1GtRecordInputTag,
        const edm::InputTag& l1GtReadoutRecordInputTag) {

    // reset the vectors before filling them
    reset(m_decisionsBeforeMask);
    reset(m_decisionsAfterMask);
    reset(m_prescaleFactors);
    reset(m_triggerMasks);
    reset(m_errorCodes);

    // initialization of actual values for each trigger
    bool decisionBeforeMaskValue = false;
    bool decisionAfterMaskValue = false;
    int prescaleFactorValue = -1;
    int triggerMaskValue = -1;
    int errorCode = -1;

    LogDebug("L1GtUtils") << std::endl;
    LogTrace("L1GtUtils") << "\nLogical expression\n  " << m_logicalExpression
            << std::endl;

    // for each trigger, if it is in the L1 menu, get the prescale factor and trigger mask

    for (size_t iTrig = 0; iTrig < m_expL1TriggersSize; ++iTrig) {

        const std::string& trigNameOrAlias = (m_expL1Triggers[iTrig]).tokenName;

        if (m_expTriggerInMenu[iTrig]) {
            errorCode = m_l1GtUtils.l1Results(iEvent, l1GtRecordInputTag,
                    l1GtReadoutRecordInputTag, trigNameOrAlias,
                    decisionBeforeMaskValue, decisionAfterMaskValue,
                    prescaleFactorValue, triggerMaskValue);

            if (errorCode != 0) {

                // error while retrieving the results
                //    for this trigger: set prescale factor to -1, trigger mask to -1

                decisionBeforeMaskValue = false;
                decisionAfterMaskValue = false;
                prescaleFactorValue = -1;
                triggerMaskValue = -1;

            }

        } else {
            // no trigger name or trigger alias in the menu, no bits: 
            //    for this trigger: set prescale factor to -1, set the error code

            decisionBeforeMaskValue = false;
            decisionAfterMaskValue = false;
            prescaleFactorValue = -1;
            triggerMaskValue = -1;
            errorCode = m_l1ConfCode + 1;

        }

        LogTrace("L1GtUtils") << "\n" << trigNameOrAlias << ":" << std::endl;

        (m_decisionsBeforeMask[iTrig]).second = decisionBeforeMaskValue;
        LogTrace("L1GtUtils") << "    decision before mask = "
                << decisionBeforeMaskValue << std::endl;

        (m_decisionsAfterMask[iTrig]).second = decisionAfterMaskValue;
        LogTrace("L1GtUtils") << "    decision after mask  = "
                << decisionAfterMaskValue << std::endl;

        (m_prescaleFactors[iTrig]).second = prescaleFactorValue;
        LogTrace("L1GtUtils") << "    prescale factor      = "
                << prescaleFactorValue << std::endl;

        (m_triggerMasks[iTrig]).second = triggerMaskValue;
        LogTrace("L1GtUtils") << "    trigger mask         = "
                << triggerMaskValue << std::endl;

        (m_errorCodes[iTrig]).second = errorCode;
        LogTrace("L1GtUtils") << "    error code           = " << errorCode
                << std::endl;

    }

    LogDebug("L1GtUtils") << std::endl;

}

const std::string L1GtUtils::EmptyString = "";
const int L1GtUtils::L1GtNotValidError = 99999;

