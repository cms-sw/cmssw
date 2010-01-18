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

            // input tag for GT DAQ record
            m_l1GtDaqReadoutRecordInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GtDaqReadoutRecordInputTag")),

            // input tag for GT lite record
            m_l1GtRecordInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GtRecordInputTag")),

            // input tag for GT object map collection
            m_l1GtObjectMapTag(parSet.getParameter<edm::InputTag>(
                    "L1GtObjectMapTag")),

            // input tag for muon collection from GMT
            m_l1GmtInputTag(parSet.getParameter<edm::InputTag>(
                    "L1GmtInputTag")),

            /// an algorithm and a condition in that algorithm to test the object maps
            m_nameAlgTechTrig(parSet.getParameter<std::string> ("AlgorithmName")),
            m_condName(parSet.getParameter<std::string> ("ConditionName"))

{
    LogDebug("L1GtAnalyzer")
            << "\n Input parameters for L1 GT test analyzer"
            << "\n   L1 GT DAQ record:            "
            << m_l1GtDaqReadoutRecordInputTag
            << "\n   L1 GT lite record:           "
            << m_l1GtRecordInputTag
            << "\n   L1 GT object map collection: "
            << m_l1GtObjectMapTag
            << "\n   Muon collection from GMT:    "
            << m_l1GmtInputTag
            << "\n   Algorithm name or alias, technical trigger name: " << m_nameAlgTechTrig
            << "\n   Condition, if a physics algorithm is requested:  " << m_condName
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

        LogDebug("L1GtUtils") << "\nL1GlobalTriggerReadoutRecord with \n  "
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

    // access L1 trigger results using public methods from L1GtUtils

    myCoutStream << "\nL1 trigger menu: \n" << m_l1GtUtils.l1TriggerMenu()
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
        myCoutStream << "Decision after trigger mask for "
                << m_nameAlgTechTrig << ":    " << decisionAfterMaskAlgTechTrig
                << std::endl;
        myCoutStream << "Decision (after trigger mask) for "
                << m_nameAlgTechTrig << ":  " << decisionAlgTechTrig
                << std::endl;

        myCoutStream << "Prescale factor for " << m_nameAlgTechTrig
                << ":                " << prescaleFactorAlgTechTrig << std::endl;

        myCoutStream << "Trigger mask for " << m_nameAlgTechTrig
                << ":                   " << triggerMaskAlgTechTrig
                << std::endl;

    } else if (iErrorCode == 1) {
        myCoutStream << "\n" << m_nameAlgTechTrig
                << " does not exist in the L1 menu "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

    } else {
        myCoutStream << "\nError: "
                << "\n  An error was encountered when retrieving decisionBeforeMask for "
                << m_nameAlgTechTrig << "\n  Error code = " << iErrorCode
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

    }

    // another method to get the trigger mask (no common errorCode)

    iErrorCode = -1;
    triggerMaskAlgTechTrig = m_l1GtUtils.triggerMask(m_nameAlgTechTrig,
            iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "Trigger mask for " << m_nameAlgTechTrig
                << "(faster method):    " << triggerMaskAlgTechTrig
                << std::endl;

    } else if (iErrorCode == 1) {
        myCoutStream << "\n" << m_nameAlgTechTrig
                << " does not exist in the L1 menu "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

    } else {
        myCoutStream << "\nError: "
                << "\n  An error was encountered when retrieving decisionBeforeMask for "
                << m_nameAlgTechTrig << "\n  Error code = " << iErrorCode
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

    }

    // sets of prescale factors and trigger masks for physics algorithms

    std::string triggerAlgTechTrig = "PhysicsAlgorithms";

    iErrorCode = -1;
    const std::vector<int>& pfSetPhysicsAlgorithms =
            m_l1GtUtils.prescaleFactorSet(iEvent, triggerAlgTechTrig,
                    iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: prescale factor set for run "
                << iEvent.run() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

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
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << std::endl;
    }

    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetPhysicsAlgorithms =
            m_l1GtUtils.triggerMaskSet(triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nPhysics algorithms: trigger mask set for run "
                << iEvent.run() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

        int iBit = -1;
        for (std::vector<unsigned int>::const_iterator
                cItBit = tmSetPhysicsAlgorithms.begin();
                cItBit != tmSetPhysicsAlgorithms.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": trigger mask = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the trigger mask set "
                << "\n  for physics algorithms, for run " << iEvent.run()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << std::endl;
    }


    // sets of prescale factors and trigger masks for technical triggers

    triggerAlgTechTrig = "TechnicalTriggers";
    iErrorCode = -1;
    const std::vector<int>& pfSetTechnicalTriggers =
            m_l1GtUtils.prescaleFactorSet(iEvent, triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: prescale factor set for run "
                << iEvent.run() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

        int iBit = -1;
        for (std::vector<int>::const_iterator cItBit =
                pfSetTechnicalTriggers.begin(); cItBit
                != pfSetTechnicalTriggers.end(); ++cItBit) {

            iBit++;
            myCoutStream << "Bit number " << std::right << std::setw(4) << iBit
                    << ": prescale factor = " << (*cItBit) << std::endl;

        }

    } else {
        myCoutStream
                << "\nError encountered when retrieving the prescale factor set "
                << "\n  for technical triggers, for run " << iEvent.run()
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << std::endl;
    }

    iErrorCode = -1;
    const std::vector<unsigned int>& tmSetTechnicalTriggers =
            m_l1GtUtils.triggerMaskSet(triggerAlgTechTrig, iErrorCode);

    if (iErrorCode == 0) {
        myCoutStream << "\nTechnical triggers: trigger mask set for run "
                << iEvent.run() << ", with L1 menu \n  "
                << m_l1GtUtils.l1TriggerMenu() << std::endl;

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
                << " with L1 menu \n  " << m_l1GtUtils.l1TriggerMenu()
                << "\n  Error code: " << iErrorCode << std::endl;
    }



    LogDebug("L1GtAnalyzer") << myCoutStream.str() << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}



// analyze: object map record
void L1GtAnalyzer::analyzeObjectMap(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    LogDebug("L1GtAnalyzer")
            << "\n**** L1GtAnalyzer::analyzeObjectMap object map record ****\n"
            << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get a handle to the object map record
    // the record can come only from emulator - no hardware ObjectMapRecord
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

    // analyze: object map record
    analyzeObjectMap(iEvent, evSetup);

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

//define this as a plug-in
DEFINE_FWK_MODULE(L1GtAnalyzer);
