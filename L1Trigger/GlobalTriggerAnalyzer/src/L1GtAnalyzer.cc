/**
 * \class L1GtAnalyzer
 * 
 * 
 * Description: test analyzer to illustrate various methods for L1 GT trigger.
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtAnalyzer.h"

// system include files
#include <memory>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// constructor(s)
L1GtAnalyzer::L1GtAnalyzer(const edm::ParameterSet& parSet) :

            // input tag for GT DAQ product
            m_l1GtDaqReadoutRecordInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GtDaqReadoutRecordInputTag")),

            // input tag for GT lite product
            m_l1GtRecordInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GtRecordInputTag")),

            // input tag for GT object map collection
            m_l1GtObjectMapTag(parSet.getParameter<edm::InputTag>(
                    "L1GtObjectMapTag")),

            // input tag for muon collection from GMT
            m_l1GmtInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GmtInputTag")),

            // input tag for L1GtTriggerMenuLite
            m_l1GtTmLInputTag(parSet.getParameter<edm::InputTag> (
                    "L1GtTmLInputTag")),

            // input tag for ConditionInEdm products
            m_condInEdmInputTag(parSet.getParameter<edm::InputTag> (
                            "CondInEdmInputTag")),

            // an algorithm and a condition in that algorithm to test the object maps
            m_nameAlgTechTrig(parSet.getParameter<std::string> ("AlgorithmName")),
            m_condName(parSet.getParameter<std::string> ("ConditionName")),
            m_bitNumber(parSet.getParameter<unsigned int> ("BitNumber")),

            m_l1GtUtilsConfiguration(parSet.getParameter<unsigned int> ("L1GtUtilsConfiguration")),
            m_l1GtTmLInputTagProv(parSet.getParameter<bool> ("L1GtTmLInputTagProv")),
            m_l1GtUtilsConfigureBeginRun(parSet.getParameter<bool> ("L1GtUtilsConfigureBeginRun"))

{
    LogDebug("L1GtAnalyzer")
            << "\n Input parameters for L1 GT test analyzer"
            << "\n   L1 GT DAQ product:            "
            << m_l1GtDaqReadoutRecordInputTag
            << "\n   L1 GT lite product:           "
            << m_l1GtRecordInputTag
            << "\n   L1 GT object map collection:  "
            << m_l1GtObjectMapTag
            << "\n   Muon collection from GMT:     "
            << m_l1GmtInputTag
            << "\n   L1 trigger menu lite product: "
            << m_l1GtTmLInputTag
            << "\n   Algorithm name or alias, technical trigger name:  " << m_nameAlgTechTrig
            << "\n   Condition, if a physics algorithm is requested:   " << m_condName
            << "\n   Bit number for an algorithm or technical trigger: " << m_bitNumber
            << "\n   Requested L1 trigger configuration: " << m_l1GtUtilsConfiguration
            << "\n   Retrieve input tag from provenance for L1 trigger menu lite in the L1GtUtils: "
            << m_l1GtTmLInputTagProv
            << "\n   Configure L1GtUtils in beginRun(...): "
            << m_l1GtUtilsConfigureBeginRun
            << " \n" << std::endl;

}

// destructor
L1GtAnalyzer::~L1GtAnalyzer() {

    // empty

}

// method called once each job just before starting event loop
void L1GtAnalyzer::beginJob()
{

    // empty

}

void L1GtAnalyzer::beginRun(const edm::Run& iRun,
        const edm::EventSetup& evSetup) {

    analyzeConditionsInRunBlock(iRun, evSetup);

    // L1GtUtils

    if (m_l1GtUtilsConfigureBeginRun) {

        //   for tests, use only one of the following methods for m_l1GtUtilsConfiguration

        bool useL1EventSetup = false;
        bool useL1GtTriggerMenuLite = false;

        switch (m_l1GtUtilsConfiguration) {
            case 0: {
                useL1EventSetup = false;
                useL1GtTriggerMenuLite = true;

            }
                break;
            case 100000: {
                useL1EventSetup = true;
                useL1GtTriggerMenuLite = true;

            }
                break;
            case 200000: {
                useL1EventSetup = true;
                useL1GtTriggerMenuLite = false;

            }
                break;
            default: {
                // do nothing
            }
                break;
        }

        if (m_l1GtTmLInputTagProv) {
            // L1GtTriggerMenuLite input tag from provenance
            m_l1GtUtils.getL1GtRunCache(iRun, evSetup, useL1EventSetup,
                    useL1GtTriggerMenuLite);

        } else {

            // L1GtTriggerMenuLite input tag given in configuration file
            m_l1GtUtils.getL1GtRunCache(iRun, evSetup, useL1EventSetup,
                    useL1GtTriggerMenuLite, m_l1GtTmLInputTag);
        }

    }

}


void L1GtAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    analyzeConditionsInLumiBlock(iLumi, evSetup);

}


// member functions

// analyze: decision and decision word
//   bunch cross in event BxInEvent = 0 - L1Accept event
void L1GtAnalyzer::analyzeDecisionReadoutRecord(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeDecisionReadoutRecord ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_l1GtDaqReadoutRecordInputTag, gtReadoutRecord);

    if (!gtReadoutRecord.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nL1GlobalTriggerReadoutRecord with \n  "
                << m_l1GtDaqReadoutRecordInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    // get Global Trigger decision and the decision word
    bool gtDecision = gtReadoutRecord->decision();
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    // print Global Trigger decision and the decision word
    edm::LogVerbatim("L1GtAnalyzer")
    << "\n GlobalTrigger decision: " << gtDecision << std::endl;

    // print via supplied "print" function (
    gtReadoutRecord->printGtDecision(myCoutStream);

    // print technical trigger word via supplied "print" function
    gtReadoutRecord->printTechnicalTrigger(myCoutStream);

    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}

// analyze: decision for a given algorithm via trigger menu
void L1GtAnalyzer::analyzeDecisionLiteRecord(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeDecisionLiteRecord ****\n"
    << std::endl;

    edm::Handle<L1GlobalTriggerRecord> gtRecord;
    iEvent.getByLabel(m_l1GtRecordInputTag, gtRecord);

    if (!gtRecord.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nL1GlobalTriggerRecord with \n  "
                << m_l1GtRecordInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;

    }

    // FIXME - un-comment when new tag for CondFormats ready
//    const DecisionWord gtDecisionWord = gtRecord->decisionWord();
//
//    edm::ESHandle<L1GtTriggerMenu> l1GtMenu;
//    evSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu) ;
//    const L1GtTriggerMenu* m_l1GtMenu = l1GtMenu.product();
//

//    bool trigResult = false;
//
//    if (m_l1GtMenu->gtTriggerResult(m_nameAlgTechTrig, gtDecisionWord,
//            trigResult)) {
//
//        edm::LogVerbatim("L1GtAnalyzer") << "\nResult for trigger "
//                << m_nameAlgTechTrig << ": " << trigResult << "\n" << std::endl;
//    } else {
//        edm::LogVerbatim("L1GtAnalyzer")
//                << "\nError retrieving result for trigger " << m_nameAlgTechTrig
//                << ": " << trigResult << "\n" << std::endl;
//
//    }

}

void L1GtAnalyzer::analyzeL1GtUtilsCore(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {


    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;


    // example to access L1 trigger results using public methods from L1GtUtils
    // all must be called after one or both of the commands retrieving the L1 configuration
    //   m_l1GtUtils.retrieveL1EventSetup(evSetup);
    //   m_l1GtUtils.retrieveL1GtTriggerMenuLite(iEvent, m_l1GtTmLInputTag);
    //
    //

    // testing which environment is used

    int iErrorCode = -1;
    int l1ConfCode = -1;

    const bool l1Conf = m_l1GtUtils.availableL1Configuration(iErrorCode, l1ConfCode);

    myCoutStream << "\nL1 configuration code: \n"
            << "\n Legend: "
            << "\n      0 - Retrieve L1 trigger configuration from L1GtTriggerMenuLite only"
            << "\n  10000     L1GtTriggerMenuLite product is valid"
            << "\n  99999     L1GtTriggerMenuLite product not valid. Error."
            << "\n"
            << "\n 100000 - Fall through: try first L1GtTriggerMenuLite; if not valid,try event setup."
            << "\n 110000     L1GtTriggerMenuLite product is valid"
            << "\n 120000     L1GtTriggerMenuLite product not valid, event setup valid."
            << "\n 199999     L1GtTriggerMenuLite product not valid, event setup not valid. Error."
            << "\n"
            << "\n 200000 - Retrieve L1 trigger configuration from event setup only."
            << "\n 210000     Event setup valid."
            << "\n 299999     Event setup not valid. Error."
            << "\n"
            << "\n 300000 - No L1 trigger configuration requested to be retrieved. Error"
            << "\n            Must call before using L1GtUtils methods: "
            << "\n                getL1GtRunCache(const edm::Event& iEvent, const edm::EventSetup& evSetup,"
            << "\n                                const bool useL1EventSetup, const bool useL1GtTriggerMenuLite)"
            << "\n"
            << std::endl;


    if (l1Conf) {
        myCoutStream << "\nL1 configuration code:" << l1ConfCode
                << "\nValid L1 trigger configuration." << std::endl;

        myCoutStream << "\nL1 trigger menu name and implementation:" << "\n"
                << m_l1GtUtils.l1TriggerMenu() << "\n"
                << m_l1GtUtils.l1TriggerMenuImplementation() << std::endl;

    } else {
        myCoutStream << "\nL1 configuration code:" << l1ConfCode
                << "\nNo valid L1 trigger configuration available."
                << "\nSee text above for error code interpretation"
                << "\nNo return here, in order to test each method, protected against configuration error."
                << std::endl;
    }



    myCoutStream
            << "\n******** Results found with input tags retrieved from provenance ******** \n"
            << std::endl;

    //
    // no input tags; for the appropriate EDM product, it will be found
    // from provenance

    // the following methods share the same error code, therefore one can check only once
    // the validity of the result

    iErrorCode = -1;

    bool decisionBeforeMaskAlgTechTrig = m_l1GtUtils.decisionBeforeMask(iEvent,
            m_nameAlgTechTrig, iErrorCode);

    bool decisionAfterMaskAlgTechTrig = m_l1GtUtils.decisionAfterMask(iEvent,
            m_nameAlgTechTrig, iErrorCode);

    bool decisionAlgTechTrig = m_l1GtUtils.decision(iEvent, m_nameAlgTechTrig,
            iErrorCode);

    int prescaleFactorAlgTechTrig = m_l1GtUtils.prescaleFactor(iEvent,
            m_nameAlgTechTrig, iErrorCode);

    int triggerMaskAlgTechTrig = m_l1GtUtils.triggerMask(iEvent,
            m_nameAlgTechTrig, iErrorCode);

    myCoutStream << "\n\nMethods:"
            << "\n  decisionBeforeMask(iEvent, m_nameAlgTechTrig, iErrorCode)"
            << "\n  decisionAfterMask(iEvent, m_nameAlgTechTrig, iErrorCode)"
            << "\n  decision(iEvent, m_nameAlgTechTrig, iErrorCode)"
            << "\n  prescaleFactor(iEvent, m_nameAlgTechTrig, iErrorCode)"
            << "\n  triggerMask(iEvent, m_nameAlgTechTrig, iErrorCode)"
            << "\n  triggerMask(m_nameAlgTechTrig,iErrorCode)"
            << "\n\n" << std::endl;


    if (iErrorCode == 0) {
        myCoutStream << "\nDecision before trigger mask for "
                << m_nameAlgTechTrig << ":   " << decisionBeforeMaskAlgTechTrig
                << std::endl;
        myCoutStream << "Decision after trigger mask for " << m_nameAlgTechTrig
                << ":    " << decisionAfterMaskAlgTechTrig << std::endl;
        myCoutStream << "Decision (after trigger mask) for "
                << m_nameAlgTechTrig << ":  " << decisionAlgTechTrig
                << std::endl;

        myCoutStream << "Prescale factor for " << m_nameAlgTechTrig
                << ":                " << prescaleFactorAlgTechTrig
                << std::endl;

        myCoutStream << "Trigger mask for " << m_nameAlgTechTrig
                << ":                   " << triggerMaskAlgTechTrig
                << std::endl;

    } else if (iErrorCode == 1) {
        myCoutStream << "\n" << m_nameAlgTechTrig
                << " does not exist in the L1 menu "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

    } else {
        myCoutStream << "\nError: "
                << "\n  An error was encountered when retrieving decision, mask and prescale factor for "
                << m_nameAlgTechTrig << "\n  L1 Menu: "
                << m_l1GtUtils.l1TriggerMenu() << "\n  Error code: "
                << iErrorCode << std::endl;

    }

    // another method to get the trigger mask (no common errorCode)

    iErrorCode = -1;
    triggerMaskAlgTechTrig = m_l1GtUtils.triggerMask(m_nameAlgTechTrig,
            iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTrigger mask for " << m_nameAlgTechTrig
                << "(faster method):    " << triggerMaskAlgTechTrig
                << std::endl;

    } else if (iErrorCode == 1) {
        myCoutStream << "\n" << m_nameAlgTechTrig
                << " does not exist in the L1 menu "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

    } else {
        myCoutStream << "\nError: "
                << "\n  An error was encountered when fast retrieving trigger mask for "
                << m_nameAlgTechTrig << "\n  L1 Menu: "
                << m_l1GtUtils.l1TriggerMenu() << "\n  Error code: "
                << iErrorCode << std::endl;

    }

    // index of the actual prescale factor set, and the actual prescale
    // factor set for algorithm triggers



    L1GtUtils::TriggerCategory trigCategory = L1GtUtils::AlgorithmTrigger;

    myCoutStream << "\nMethods:"
            << "\n  prescaleFactorSetIndex(iEvent, trigCategory, iErrorCode)"
            << "\n  prescaleFactorSet(iEvent, trigCategory,iErrorCode)\n"
            << std::endl;

    iErrorCode = -1;
    const int pfSetIndexAlgorithmTrigger = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream
                << "\nAlgorithm triggers: index for prescale factor set = "
                << pfSetIndexAlgorithmTrigger << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;


    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for algorithm triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetAlgorithmTrigger =
            m_l1GtUtils.prescaleFactorSet(iEvent, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nAlgorithm triggers: prescale factor set index = "
                << pfSetIndexAlgorithmTrigger << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetAlgorithmTrigger.begin(); cItBit
                != pfSetAlgorithmTrigger.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for algorithm triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    // the actual trigger mask set for algorithm triggers

    myCoutStream << "\nMethod:"
            << "\n  triggerMaskSet(trigCategory, iErrorCode)"
            << std::endl;

    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetAlgorithmTrigger =
            m_l1GtUtils.triggerMaskSet(trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nAlgorithm triggers: trigger mask set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<unsigned int>::const_iterator cItBit =
                tmSetAlgorithmTrigger.begin(); cItBit
                != tmSetAlgorithmTrigger.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": trigger mask = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the trigger mask set "
                << "\n  for algorithm triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }




    // index of the actual prescale factor set, and the actual prescale
    // factor set for technical triggers

    trigCategory = L1GtUtils::TechnicalTrigger;

    myCoutStream << "\nMethods:"
            << "\n  prescaleFactorSetIndex(iEvent, trigCategory, iErrorCode)"
            << "\n  prescaleFactorSet(iEvent, trigCategory,iErrorCode)\n"
            << std::endl;

    iErrorCode = -1;
    const int pfSetIndexTechnicalTrigger = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream
                << "\nTechnical triggers: index for prescale factor set = "
                << pfSetIndexTechnicalTrigger << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\nMethod: prescaleFactorSetIndex(iEvent, trigCategory, iErrorCode)\n"
                << std::endl;

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetTechnicalTrigger =
            m_l1GtUtils.prescaleFactorSet(iEvent, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: prescale factor set index = "
                << pfSetIndexTechnicalTrigger << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\nMethod: prescaleFactorSet(iEvent, trigCategory,iErrorCode)\n"
                << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetTechnicalTrigger.begin(); cItBit
                != pfSetTechnicalTrigger.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    // the actual trigger mask set for technical triggers

    myCoutStream << "\nMethod:"
            << "\n  triggerMaskSet(trigCategory, iErrorCode)"
            << std::endl;

    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetTechnicalTrigger =
            m_l1GtUtils.triggerMaskSet(trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: trigger mask set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<unsigned int>::const_iterator cItBit =
                tmSetTechnicalTrigger.begin(); cItBit
                != tmSetTechnicalTrigger.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": trigger mask = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the trigger mask set "
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }





    //
    // same methods as above, but with input tag given explicitly, allowing to select
    // the EDM products used to get the results

    myCoutStream
            << "\n******** Results found with input tags provided in the configuration file ******** \n"
            << "\n  L1GlobalTriggerRecord: " << m_l1GtRecordInputTag
            << "\n  L1GlobalTriggerReadoutRecord: "
            << m_l1GtDaqReadoutRecordInputTag << std::endl;


    // the following methods share the same error code, therefore one can check only once
    // the validity of the result

    iErrorCode = -1;

    bool decisionBeforeMaskAlgTechTrigITag = m_l1GtUtils.decisionBeforeMask(iEvent,
            m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    bool decisionAfterMaskAlgTechTrigITag = m_l1GtUtils.decisionAfterMask(iEvent,
            m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    bool decisionAlgTechTrigITag = m_l1GtUtils.decision(iEvent,
            m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    int prescaleFactorAlgTechTrigITag = m_l1GtUtils.prescaleFactor(iEvent,
            m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    int triggerMaskAlgTechTrigITag = m_l1GtUtils.triggerMask(iEvent,
            m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    myCoutStream << "\n\nMethods:"
            << "\n  decisionBeforeMask(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, m_nameAlgTechTrig, iErrorCode)"
            << "\n  decisionAfterMask(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, m_nameAlgTechTrig, iErrorCode)"
            << "\n  decision(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, m_nameAlgTechTrig, iErrorCode)"
            << "\n  prescaleFactor(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, m_nameAlgTechTrig, iErrorCode)"
            << "\n  triggerMask(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, m_nameAlgTechTrig, iErrorCode)"
            << "\n\n"
            << std::endl;


    if (iErrorCode == 0) {
        myCoutStream << "\nDecision before trigger mask for "
                << m_nameAlgTechTrig << ":   " << decisionBeforeMaskAlgTechTrigITag
                << std::endl;
        myCoutStream << "Decision after trigger mask for " << m_nameAlgTechTrig
                << ":    " << decisionAfterMaskAlgTechTrigITag << std::endl;
        myCoutStream << "Decision (after trigger mask) for "
                << m_nameAlgTechTrig << ":  " << decisionAlgTechTrigITag
                << std::endl;

        myCoutStream << "Prescale factor for " << m_nameAlgTechTrig
                << ":                " << prescaleFactorAlgTechTrigITag
                << std::endl;

        myCoutStream << "Trigger mask for " << m_nameAlgTechTrig
                << ":                   " << triggerMaskAlgTechTrigITag
                << std::endl;

    } else if (iErrorCode == 1) {
        myCoutStream << "\n" << m_nameAlgTechTrig
                << " does not exist in the L1 menu "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

    } else {
        myCoutStream << "\nError: "
                << "\n  An error was encountered when retrieving decision, mask and prescale factor for "
                << m_nameAlgTechTrig << "\n  L1 Menu: "
                << m_l1GtUtils.l1TriggerMenu() << "\n  Error code: "
                << iErrorCode << std::endl;

    }


    // index of the actual prescale factor set, and the actual prescale
    // factor set for algorithm triggers



    trigCategory = L1GtUtils::AlgorithmTrigger;

    myCoutStream << "\nMethods:"
            << "\n  prescaleFactorSetIndex(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, trigCategory, iErrorCode)"
            << "\n  prescaleFactorSet(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, trigCategory,iErrorCode)\n"
            << std::endl;

    iErrorCode = -1;
    const int pfSetIndexAlgorithmTriggerITag = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream
                << "\nAlgorithm triggers: index for prescale factor set = "
                << pfSetIndexAlgorithmTriggerITag << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;


    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for algorithm triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetAlgorithmTriggerITag =
            m_l1GtUtils.prescaleFactorSet(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nAlgorithm triggers: prescale factor set index = "
                << pfSetIndexAlgorithmTriggerITag << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetAlgorithmTriggerITag.begin(); cItBit
                != pfSetAlgorithmTriggerITag.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for algorithm triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }






    // index of the actual prescale factor set, and the actual prescale
    // factor set for technical triggers

    trigCategory = L1GtUtils::TechnicalTrigger;

    myCoutStream << "\nMethods:"
            << "\n  prescaleFactorSetIndex(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, trigCategory, iErrorCode)"
            << "\n  prescaleFactorSet(iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag, trigCategory,iErrorCode)\n"
            << std::endl;

    iErrorCode = -1;
    const int pfSetIndexTechnicalTriggerITag = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream
                << "\nTechnical triggers: index for prescale factor set = "
                << pfSetIndexTechnicalTriggerITag << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetTechnicalTriggerITag =
            m_l1GtUtils.prescaleFactorSet(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, trigCategory, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: prescale factor set index = "
                << pfSetIndexTechnicalTriggerITag << "\nfor run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetTechnicalTriggerITag.begin(); cItBit
                != pfSetTechnicalTriggerITag.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }



    //
    // dump the stream in some Log tag (LogDebug here)


    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}

void L1GtAnalyzer::analyzeL1GtUtilsMenuLite(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeL1GtUtilsMenuLite ****\n"
            << std::endl;

    // before accessing any result from L1GtUtils, one must retrieve and cache
    // the L1GtTriggerMenuLite product
    // add this call in the analyze / produce / filter method of your
    // analyzer / producer / filter

    bool useL1EventSetup = false;
    bool useL1GtTriggerMenuLite = true;

    if (m_l1GtTmLInputTagProv) {

        // input tag for L1GtTriggerMenuLite retrieved from provenance
        m_l1GtUtils.getL1GtRunCache(iEvent, evSetup, useL1EventSetup,
                useL1GtTriggerMenuLite);

    } else {

        // input tag for L1GtTriggerMenuLite explicitly given
        m_l1GtUtils.getL1GtRunCache(iEvent, evSetup, useL1EventSetup,
                useL1GtTriggerMenuLite, m_l1GtTmLInputTag);
    }

    analyzeL1GtUtilsCore(iEvent, evSetup);

}

void L1GtAnalyzer::analyzeL1GtUtilsEventSetup(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeL1GtUtilsEventSetup ****\n"
            << std::endl;

    // before accessing any result from L1GtUtils, one must retrieve and cache
    // the L1 trigger event setup
    // add this call in the analyze / produce / filter method of your
    // analyzer / producer / filter

    bool useL1EventSetup = true;
    bool useL1GtTriggerMenuLite = false;

    m_l1GtUtils.getL1GtRunCache(iEvent, evSetup, useL1EventSetup,
            useL1GtTriggerMenuLite);

    analyzeL1GtUtilsCore(iEvent, evSetup);

}

void L1GtAnalyzer::analyzeL1GtUtils(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeL1GtUtils: fall-through case ****\n"
            << std::endl;

    // before accessing any result from L1GtUtils, one must retrieve and cache
    // the L1 trigger event setup and the L1GtTriggerMenuLite product
    // add this call in the analyze / produce / filter method of your
    // analyzer / producer / filter

    bool useL1EventSetup = true;
    bool useL1GtTriggerMenuLite = true;

    if (m_l1GtTmLInputTagProv) {

        // input tag for L1GtTriggerMenuLite retrieved from provenance
        m_l1GtUtils.getL1GtRunCache(iEvent, evSetup, useL1EventSetup,
                useL1GtTriggerMenuLite);

    } else {

        // input tag for L1GtTriggerMenuLite explicitly given
        m_l1GtUtils.getL1GtRunCache(iEvent, evSetup, useL1EventSetup,
                useL1GtTriggerMenuLite, m_l1GtTmLInputTag);

    }

    analyzeL1GtUtilsCore(iEvent, evSetup);

}

// analyze: object map product
void L1GtAnalyzer::analyzeObjectMap(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeObjectMap object map product ****\n"
            << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get a handle to the object map product
    // the product can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_l1GtObjectMapTag, gtObjectMapRecord);

    if (!gtObjectMapRecord.isValid()) {
        LogDebug("L1GtAnalyzer")
                << "\nWarning: L1GlobalTriggerObjectMapRecord with input tag "
                << m_l1GtObjectMapTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    // get all object maps
    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
            gtObjectMapRecord->gtObjectMap();

    // print every object map via the implemented print
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator it =
            objMapVec.begin(); it != objMapVec.end(); ++it) {

        (*it).print(myCoutStream);
    }

    //
    const CombinationsInCond* comb = gtObjectMapRecord->getCombinationsInCond(
            m_nameAlgTechTrig, m_condName);

    // number of combinations
    if (comb != 0) {
        myCoutStream << "\n  Number of combinations passing ("
                << m_nameAlgTechTrig << ", " << m_condName << "): "
                << comb->size() << std::endl;
    } else {
        myCoutStream << "\n  No combination passes (" << m_nameAlgTechTrig
                << ", " << m_condName << ") " << std::endl;

    }

    // condition result
    const bool result = gtObjectMapRecord->getConditionResult(
            m_nameAlgTechTrig, m_condName);

    myCoutStream << "\n  Result for condition " << m_condName
            << " in algorithm " << m_nameAlgTechTrig << ": " << result
            << std::endl;

    // print all the stuff if at LogDebug level
    LogDebug("L1GtAnalyzer")
            << "Test gtObjectMapRecord in L1GlobalTrigger \n\n"
            << myCoutStream.str() << "\n\n" << std::endl;
    myCoutStream.str("");
    myCoutStream.clear();

}

// analyze: usage of L1GtTriggerMenuLite
void L1GtAnalyzer::analyzeL1GtTriggerMenuLite(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeL1GtTriggerMenuLite ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get Run Data - the same code can be run in beginRun, with getByLabel from edm::Run
    const edm::Run& iRun = iEvent.getRun();


    // get L1GtTriggerMenuLite
    edm::Handle<L1GtTriggerMenuLite> triggerMenuLite;
    iRun.getByLabel(m_l1GtTmLInputTag, triggerMenuLite);

    if (!triggerMenuLite.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nL1GtTriggerMenuLite with \n  "
                << m_l1GtTmLInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    // print via supplied "print" function
    myCoutStream << (*triggerMenuLite);

    // test the individual methods

    const std::string& triggerMenuInterface =
            triggerMenuLite->gtTriggerMenuInterface();
    const std::string& triggerMenuName = triggerMenuLite->gtTriggerMenuName();
    const std::string& triggerMenuImplementation =
            triggerMenuLite->gtTriggerMenuImplementation();
    const std::string& scaleDbKey = triggerMenuLite->gtScaleDbKey();

    const L1GtTriggerMenuLite::L1TriggerMap& algorithmMap = triggerMenuLite->gtAlgorithmMap();
    const L1GtTriggerMenuLite::L1TriggerMap& algorithmAliasMap =
            triggerMenuLite->gtAlgorithmAliasMap();
    const L1GtTriggerMenuLite::L1TriggerMap& technicalTriggerMap =
            triggerMenuLite->gtTechnicalTriggerMap();

    const std::vector<unsigned int>& triggerMaskAlgoTrig =
            triggerMenuLite->gtTriggerMaskAlgoTrig();
    const std::vector<unsigned int>& triggerMaskTechTrig =
            triggerMenuLite->gtTriggerMaskTechTrig();

    const std::vector<std::vector<int> >& prescaleFactorsAlgoTrig =
            triggerMenuLite->gtPrescaleFactorsAlgoTrig();
    const std::vector<std::vector<int> >& prescaleFactorsTechTrig =
            triggerMenuLite->gtPrescaleFactorsTechTrig();

    // print in the same format as in L1GtTriggerMenuLite definition

    size_t nrDefinedAlgo = algorithmMap.size();
    size_t nrDefinedTech = technicalTriggerMap.size();

    // header for printing algorithms

    myCoutStream << "\n   ********** L1 Trigger Menu - printing   ********** \n"
    << "\nL1 Trigger Menu Interface: " << triggerMenuInterface
    << "\nL1 Trigger Menu Name:      " << triggerMenuName
    << "\nL1 Trigger Menu Implementation: " << triggerMenuImplementation
    << "\nAssociated Scale DB Key: " << scaleDbKey << "\n\n"
    << "\nL1 Physics Algorithms: " << nrDefinedAlgo << " algorithms defined." << "\n\n"
    << "Bit Number "
    << std::right << std::setw(35) << "Algorithm Name" << "  "
    << std::right << std::setw(35) << "Algorithm Alias" << "  "
    << std::right << std::setw(12) << "Trigger Mask";
    for (unsigned iSet = 0; iSet < prescaleFactorsAlgoTrig.size(); iSet++) {
        myCoutStream << std::right << std::setw(10) << "PF Set "
               << std::right << std::setw(2)  << iSet;
    }

    myCoutStream << std::endl;


    for (L1GtTriggerMenuLite::CItL1Trig itTrig = algorithmMap.begin(); itTrig
            != algorithmMap.end(); itTrig++) {

        const unsigned int bitNumber = itTrig->first;
        const std::string& aName = itTrig->second;

        std::string aAlias;
        L1GtTriggerMenuLite::CItL1Trig itAlias = algorithmAliasMap.find(bitNumber);
        if (itAlias != algorithmAliasMap.end()) {
            aAlias = itAlias->second;
        }

        myCoutStream << std::setw(6) << bitNumber << "     "
            << std::right << std::setw(35) << aName << "  "
            << std::right << std::setw(35) << aAlias << "  "
            << std::right << std::setw(12) << triggerMaskAlgoTrig[bitNumber];
        for (unsigned iSet = 0; iSet < prescaleFactorsAlgoTrig.size(); iSet++) {
            myCoutStream << std::right << std::setw(12) << prescaleFactorsAlgoTrig[iSet][bitNumber];
        }

        myCoutStream << std::endl;
    }

    myCoutStream << "\nL1 Technical Triggers: " << nrDefinedTech
            << " technical triggers defined." << "\n\n" << std::endl;
    if (nrDefinedTech) {
        myCoutStream
            << std::right << std::setw(6) << "Bit Number "
            << std::right << std::setw(45) << " Technical trigger name " << "  "
            << std::right << std::setw(12) << "Trigger Mask";
        for (unsigned iSet = 0; iSet < prescaleFactorsTechTrig.size(); iSet++) {
            myCoutStream << std::right << std::setw(10) << "PF Set "
                    << std::right << std::setw(2) << iSet;
        }

        myCoutStream << std::endl;
    }

    for (L1GtTriggerMenuLite::CItL1Trig itTrig = technicalTriggerMap.begin(); itTrig
            != technicalTriggerMap.end(); itTrig++) {

        unsigned int bitNumber = itTrig->first;
        std::string aName = itTrig->second;

        myCoutStream << std::setw(6) << bitNumber << "       "
        << std::right << std::setw(45) << aName
        << std::right << std::setw(12) << triggerMaskTechTrig[bitNumber];
        for (unsigned iSet = 0; iSet < prescaleFactorsTechTrig.size(); iSet++) {
            myCoutStream << std::right << std::setw(12) << prescaleFactorsTechTrig[iSet][bitNumber];
        }

        myCoutStream << std::endl;

    }

    // individual methods

    int errorCode = -1;
    const std::string* algorithmAlias = triggerMenuLite->gtAlgorithmAlias(
            m_bitNumber, errorCode);
    if (errorCode) {
        myCoutStream
                << "\nError code retrieving alias for algorithm with bit number "
                << m_bitNumber << ": " << errorCode << std::endl;
    } else {
        myCoutStream << "\nAlias for algorithm with bit number " << m_bitNumber
                << ": " << (*algorithmAlias) << std::endl;
    }

    errorCode = -1;
    const std::string* algorithmName = triggerMenuLite->gtAlgorithmName(
            m_bitNumber, errorCode);
    if (errorCode) {
        myCoutStream
                << "\nError code retrieving name for algorithm with bit number "
                << m_bitNumber << ": " << errorCode << std::endl;
    } else {
        myCoutStream << "\nName for algorithm with bit number " << m_bitNumber
                << ": " << (*algorithmName) << std::endl;
    }

    errorCode = -1;
    const std::string* techTrigName = triggerMenuLite->gtTechTrigName(
            m_bitNumber, errorCode);
    if (errorCode) {
        myCoutStream
                << "\nError code retrieving name for technical trigger with bit number "
                << m_bitNumber << ": " << errorCode << std::endl;
    } else {
        myCoutStream << "\nName for technical trigger with bit number "
                << m_bitNumber << ": " << (*techTrigName) << std::endl;
    }

    errorCode = -1;
    const unsigned int bitNumber = triggerMenuLite->gtBitNumber(
            m_nameAlgTechTrig, errorCode);
    if (errorCode) {
        myCoutStream
                << "\nError code retrieving bit number for algorithm/technical trigger "
                << m_nameAlgTechTrig << ": " << errorCode << std::endl;
    } else {
        myCoutStream << "\nBit number for algorithm/technical trigger "
                << m_nameAlgTechTrig << ": " << bitNumber << std::endl;
    }

    // not tested
    //errorCode = -1;
    //const bool triggerMenuLite->gtTriggerResult( m_nameAlgTechTrig,
    //        const std::vector<bool>& decWord,  errorCode);


    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}

// analyze: usage of ConditionsInEdm
void L1GtAnalyzer::analyzeConditionsInRunBlock(const edm::Run& iRun,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeConditionsInRunBlock ****\n"
            << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get ConditionsInRunBlock
    edm::Handle<edm::ConditionsInRunBlock> condInRunBlock;
    iRun.getByLabel(m_condInEdmInputTag, condInRunBlock);

    if (!condInRunBlock.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nConditionsInRunBlock with \n  "
                << m_condInEdmInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    const boost::uint16_t beamModeVal = condInRunBlock->beamMode;
    const boost::uint16_t beamMomentumVal = condInRunBlock->beamMomentum;
    const boost::uint32_t lhcFillNumberVal = condInRunBlock->lhcFillNumber;

    // print via supplied "print" function
    myCoutStream << "\nLHC quantities in run " << iRun.run()
            << "\n  Beam Mode = " << beamModeVal
            << "\n  Beam Momentum = " << beamMomentumVal << " GeV"
            << "\n  LHC Fill Number = " << lhcFillNumberVal
            << std::endl;

    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}


void L1GtAnalyzer::analyzeConditionsInLumiBlock(
        const edm::LuminosityBlock& iLumi, const edm::EventSetup& evSetup) {
    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeConditionsInLumiBlock ****\n"
            << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get ConditionsInLumiBlock
    edm::Handle<edm::ConditionsInLumiBlock> condInLumiBlock;
    iLumi.getByLabel(m_condInEdmInputTag, condInLumiBlock);

    if (!condInLumiBlock.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nConditionsInLumiBlock with \n  "
                << m_condInEdmInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    const boost::uint32_t totalIntensityBeam1Val =
            condInLumiBlock->totalIntensityBeam1;
    const boost::uint32_t totalIntensityBeam2Val =
            condInLumiBlock->totalIntensityBeam2;

    myCoutStream << "\nLHC quantities in luminosity section "

            << iLumi.luminosityBlock() << " from run " << iLumi.run()
            << "\n  Total Intensity Beam 1 (Integer × 10E10 charges)  = "
            << totalIntensityBeam1Val
            << "\n  Total Intensity Beam 2 (Integer × 10E10 charges)  = "
            << totalIntensityBeam2Val << std::endl;

    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}

void L1GtAnalyzer::analyzeConditionsInEventBlock(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {
    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get ConditionsInEventBlock
    edm::Handle<edm::ConditionsInEventBlock> condInEventBlock;
    iEvent.getByLabel(m_condInEdmInputTag, condInEventBlock);

    if (!condInEventBlock.isValid()) {

        LogDebug("L1GtAnalyzer") << "\nConditionsInEventBlock with \n  "
                << m_condInEdmInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nExit the method.\n" << std::endl;

        return;
    }

    const boost::uint16_t bstMasterStatusVal =
            condInEventBlock->bstMasterStatus;
    const boost::uint32_t turnCountNumberVal =
            condInEventBlock->turnCountNumber;

    myCoutStream << "\nLHC quantities in event " << iEvent.id().event()
            << " from luminosity section " << iEvent.luminosityBlock()
            << " from run " << iEvent.run() << "\n  BST Master Status = "
            << bstMasterStatusVal << "\n  Turn count number = "
            << turnCountNumberVal << std::endl;

    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}

// analyze each event: event loop
void L1GtAnalyzer::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    // analyze: usage of ConditionsInEdm
    analyzeConditionsInEventBlock(iEvent, evSetup);

    // analyze: decision and decision word
    //   bunch cross in event BxInEvent = 0 - L1Accept event
    analyzeDecisionReadoutRecord(iEvent, evSetup);

    // analyze: decision for a given algorithm via trigger menu
    analyzeDecisionLiteRecord(iEvent, evSetup);

    // analyze: decision for a given algorithm using L1GtUtils functions
    //   for tests, use only one of the following methods

    switch (m_l1GtUtilsConfiguration) {
        case 0: {
            analyzeL1GtUtilsMenuLite(iEvent, evSetup);

        }
            break;
        case 100000: {
            analyzeL1GtUtils(iEvent, evSetup);

        }
            break;
        case 200000: {
            analyzeL1GtUtilsEventSetup(iEvent, evSetup);

        }
            break;
        default: {
            // do nothing
        }
            break;
    }


    // analyze: object map product
    analyzeObjectMap(iEvent, evSetup);

    // analyze: L1GtTriggerMenuLite
    analyzeL1GtTriggerMenuLite(iEvent, evSetup);

}

// end section
void L1GtAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    // empty

}
void L1GtAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {

    // empty

}

// method called once each job just after ending the event loop
void L1GtAnalyzer::endJob() {

    // empty

}

