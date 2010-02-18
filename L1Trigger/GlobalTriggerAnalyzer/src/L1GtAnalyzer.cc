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

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "FWCore/Framework/interface/ESHandle.h"

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

            // an algorithm and a condition in that algorithm to test the object maps
            m_nameAlgTechTrig(parSet.getParameter<std::string> ("AlgorithmName")),
            m_condName(parSet.getParameter<std::string> ("ConditionName")),
            m_bitNumber(parSet.getParameter<unsigned int> ("BitNumber"))

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
            << " \n" << std::endl;

}

// destructor
L1GtAnalyzer::~L1GtAnalyzer() {

    // empty

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
                << "\nExit the method." << std::endl;

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
                << "\nExit the method." << std::endl;

        return;

    }

    const DecisionWord gtDecisionWord = gtRecord->decisionWord();

    edm::ESHandle<L1GtTriggerMenu> l1GtMenu;
    evSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu) ;
    const L1GtTriggerMenu* m_l1GtMenu = l1GtMenu.product();

    const bool algResult = m_l1GtMenu->gtAlgorithmResult(m_nameAlgTechTrig,
            gtDecisionWord);

    edm::LogVerbatim("L1GtAnalyzer") << "\nResult for algorithm " << m_nameAlgTechTrig
            << ": " << algResult << "\n" << std::endl;

}

void L1GtAnalyzer::analyzeL1GtUtils(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeL1GtUtils ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // before accessing any result from L1GtUtils, one must retrieve and cache
    // the L1 trigger event setup
    // add this call in the analyze / produce / filter method of your
    // analyzer / producer / filter

    m_l1GtUtils.retrieveL1EventSetup(evSetup);

    //
    //
    // access L1 trigger results using public methods from L1GtUtils

    //
    // no input tag; for the appropriate EDM product, it will be found
    // from provenance

    myCoutStream << "\nL1 trigger menu: \n" << m_l1GtUtils.l1TriggerMenu()
            << std::endl;

    myCoutStream
            << "\n******** Results found with input tags retrieved from provenance ******** \n"
            << std::endl;


    // the following methods share the same error code, therefore one can check only once
    // the validity of the result

    int iErrorCode = -1;

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
    // factor set for physics algorithms

    std::string triggerAlgTechTrig = "PhysicsAlgorithms";

    iErrorCode = -1;
    const int pfSetIndexPhysicsAlgorithms = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: index for prescale factor set "
                << pfSetIndexPhysicsAlgorithms << " for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu() << "\n"
                << std::endl;

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for physics algorithms, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }


    iErrorCode = -1;
    const std::vector<int>& pfSetPhysicsAlgorithms =
            m_l1GtUtils.prescaleFactorSet(iEvent, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: prescale factor set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetPhysicsAlgorithms.begin(); cItBit
                != pfSetPhysicsAlgorithms.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for physics algorithms, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode  << "\n" << std::endl;
    }

    // the actual trigger mask set for physics algorithms

    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetPhysicsAlgorithms =
            m_l1GtUtils.triggerMaskSet(triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: trigger mask set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<unsigned int>::const_iterator cItBit =
                tmSetPhysicsAlgorithms.begin(); cItBit
                != tmSetPhysicsAlgorithms.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": trigger mask = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the trigger mask set "
                << "\n  for physics algorithms, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }


    // index of the actual prescale factor set, and the actual prescale
    // factor set for technical triggers

    triggerAlgTechTrig = "TechnicalTriggers";

    iErrorCode = -1;
    const int pfSetIndexTechnicalTriggers = m_l1GtUtils.prescaleFactorSetIndex(
            iEvent, triggerAlgTechTrig, iErrorCode);


    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: index for prescale factor set "
                << pfSetIndexTechnicalTriggers << " for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << ", with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu() << "\n"
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
    const std::vector<int>& pfSetTechnicalTriggers =
            m_l1GtUtils.prescaleFactorSet(iEvent, triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: prescale factor set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator
                cItBit = pfSetTechnicalTriggers.begin();
                cItBit != pfSetTechnicalTriggers.end(); ++cItBit) {

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


    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetTechnicalTriggers =
            m_l1GtUtils.triggerMaskSet(triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: trigger mask set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<unsigned int>::const_iterator
                cItBit = tmSetTechnicalTriggers.begin();
                cItBit != tmSetTechnicalTriggers.end(); ++cItBit) {

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
            << "\n******** Results found input tags provided in the configuration file ******** \n"
            << "\n  L1GlobalTriggerRecord: " << m_l1GtRecordInputTag
            << "\n  L1GlobalTriggerReadoutRecord: "
            << m_l1GtDaqReadoutRecordInputTag << std::endl;

    // the following methods share the same error code, therefore one can check only once
    // the validity of the result

    iErrorCode = -1;

    bool decisionBeforeMaskAlgTechTrigITag = m_l1GtUtils.decisionBeforeMask(
            iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
            m_nameAlgTechTrig, iErrorCode);

    bool decisionAfterMaskAlgTechTrigITag = m_l1GtUtils.decisionAfterMask(
            iEvent, m_l1GtRecordInputTag, m_l1GtDaqReadoutRecordInputTag,
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

    if (iErrorCode == 0) {
        myCoutStream << "\nDecision before trigger mask for "
                << m_nameAlgTechTrig << ":   "
                << decisionBeforeMaskAlgTechTrigITag << std::endl;
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
    // factor set for physics algorithms

    triggerAlgTechTrig = "PhysicsAlgorithms";

    iErrorCode = -1;
    const int pfSetIndexPhysicsAlgorithmsITag =
            m_l1GtUtils.prescaleFactorSetIndex(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: index for prescale factor set "
                << pfSetIndexPhysicsAlgorithmsITag << " for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for physics algorithms, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetPhysicsAlgorithmsITag =
            m_l1GtUtils.prescaleFactorSet(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: prescale factor set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator
                cItBit = pfSetPhysicsAlgorithmsITag.begin();
                cItBit != pfSetPhysicsAlgorithmsITag.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for physics algorithms, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    // index of the actual prescale factor set, and the actual prescale
    // factor set for technical triggers

    triggerAlgTechTrig = "TechnicalTriggers";

    iErrorCode = -1;
    const int pfSetIndexTechnicalTriggersITag =
            m_l1GtUtils.prescaleFactorSetIndex(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: index for prescale factor set "
                << pfSetIndexTechnicalTriggersITag << " for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set index"
                << "\n  for technical triggers, for run " << iEvent.run()
                << ", luminosity block " << iEvent.luminosityBlock()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << "\n" << std::endl;
    }

    iErrorCode = -1;
    const std::vector<int>& pfSetTechnicalTriggersITag =
            m_l1GtUtils.prescaleFactorSet(iEvent, m_l1GtRecordInputTag,
                    m_l1GtDaqReadoutRecordInputTag, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: prescale factor set for run "
                << iEvent.run() << ", luminosity block "
                << iEvent.luminosityBlock() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << "\n" << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator
                cItBit = pfSetTechnicalTriggersITag.begin();
                cItBit != pfSetTechnicalTriggersITag.end(); ++cItBit) {

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
                << "\nExit the method." << std::endl;

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
                << "\nExit the method." << std::endl;

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


// analyze each event: event loop
void L1GtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // analyze: decision and decision word
    //   bunch cross in event BxInEvent = 0 - L1Accept event
    analyzeDecisionReadoutRecord(iEvent, evSetup);
    
    // analyze: decision for a given algorithm via trigger menu
    analyzeDecisionLiteRecord(iEvent, evSetup);
    
    // analyze: decision for a given algorithm using L1GtUtils functions
    analyzeL1GtUtils(iEvent, evSetup);

    // analyze: object map product
    analyzeObjectMap(iEvent, evSetup);

    // analyze: L1GtTriggerMenuLite
    analyzeL1GtTriggerMenuLite(iEvent, evSetup);

}


// method called once each job just before starting event loop
void L1GtAnalyzer::beginJob()
{

    // empty

}

// method called once each job just after ending the event loop
void L1GtAnalyzer::endJob()
{

    // empty

}

