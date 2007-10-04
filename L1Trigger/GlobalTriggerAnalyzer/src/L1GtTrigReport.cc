/**
 * \class L1GtTrigReport
 * 
 * 
 * Description: L1 Trigger report.  
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"

// system include files
#include <memory>

#include <iostream>
#include <iomanip>

#include<map>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructor(s)
L1GtTrigReport::L1GtTrigReport(const edm::ParameterSet& pSet)
{

    // input tag for GT DAQ record
    m_l1GtDaqInputTag = pSet.getParameter<edm::InputTag>("L1GtDaqInputTag");

    LogDebug("L1GtTrigReport")
    << "\nInput tag for L1 GT DAQ record: "
    << m_l1GtDaqInputTag.label() << " \n"
    << std::endl;

    // inputTag for L1 Global Trigger object maps
    m_l1GtObjectMapTag = pSet.getParameter<edm::InputTag>("L1GtObjectMapTag");

    LogDebug("L1GtTrigReport")
    << "\nInput tag for L1 GT object maps: "
    << m_l1GtObjectMapTag.label() << " \n"
    << std::endl;

    // print verbosity
    m_printVerbosity = pSet.getUntrackedParameter<int>("PrintVerbosity", 0);

    LogDebug("L1GtTrigReport")
    << "\nPrint verbosity level: "
    << m_printVerbosity << " \n"
    << std::endl;

    // print output
    m_printOutput = pSet.getUntrackedParameter<int>("PrintOutput", 0);

    LogDebug("L1GtTrigReport")
    << "\nPrint output: "
    << m_printOutput << " \n"
    << std::endl;


    // initialize counters

    // number of events processed
    m_totalEvents = 0;

    // number of events with error (EDProduct[s] not found)
    m_nErrors = 0;

    /// number of events accepted by any of the L1 algorithm
    m_globalAccepts = 0;

    // TODO FIXME temporary, until L1 trigger menu implemented as EventSetup
    m_algoMap = false;

}

// destructor
L1GtTrigReport::~L1GtTrigReport()
{

    //empty

}

// member functions


// method called once each job just before starting event loop
void L1GtTrigReport::beginJob(const edm::EventSetup& evSetup)
{

    // empty

}

// analyze each event
void L1GtTrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // TODO FIXME temporary, until L1 trigger menu implemented as EventSetup
    if ( !m_algoMap ) {

        // get map from algorithm names to algorithm bits
        getAlgoMap(iEvent, evSetup);

        m_algoMap = true;

    }

    //
    m_totalEvents++;

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    try {
        iEvent.getByLabel(m_l1GtDaqInputTag.label(), gtReadoutRecord);
    } catch (...) {

        m_nErrors++;

        LogDebug("L1GtTrigReport")
        << "L1GlobalTriggerReadoutRecord with input tag " << m_l1GtDaqInputTag.label()
        << "not found.\n\n"
        << std::endl;

        return;

    }

    // get Global Trigger decision and the decision word
    bool gtDecision = gtReadoutRecord->decision();
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    if (gtDecision) {
        m_globalAccepts++;
    }

    // get prescale factos and trigger mask
    edm::ESHandle< L1GtPrescaleFactors > l1GtPF ;
    evSetup.get< L1GtPrescaleFactorsRcd >().get( l1GtPF ) ;

    edm::ESHandle< L1GtTriggerMask > l1GtTM ;
    evSetup.get< L1GtTriggerMaskRcd >().get( l1GtTM ) ;

    // sum per algorithm
    // TODO  fix when trigger menu available as EventSetup
    // temporary: use algorithm names and assume that algorithm names
    // are unique over various trigger menus
    // TODO treat correctly empty bits when menu is EventSetup

    const unsigned int numberTriggerBits = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    typedef std::map<std::string, int>::value_type valType;

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {

        bool algoResult = gtDecisionWord[iBit];
        std::string aName = m_algoBitToName[iBit];


        // increase the counter for that algorithm
        if ( m_nAlgoAccepts.count(aName) ) {
            if (aName != "") {
                if (algoResult) {
                    m_nAlgoAccepts[aName] += 1;
                } else {
                    m_nAlgoRejects[aName] += 1;
                }
            }
        } else {

            int presFactor = l1GtPF->gtPrescaleFactors().at(iBit);
            unsigned int trigMask = l1GtTM->gtTriggerMask().at(iBit);

            // TODO works properly for a set of factors only! FIXME
            m_prescaleFactor.insert(valType(aName, presFactor));
            m_triggerMask.insert(valType(aName, trigMask));

            if (aName == "") {
                m_nAlgoAccepts.insert(valType(aName, 0));
                m_nAlgoRejects.insert(valType(aName, 0));
            } else {
                if (algoResult) {
                    m_nAlgoAccepts.insert(valType(aName, 1));
                    m_nAlgoRejects.insert(valType(aName, 0));
                } else {
                    m_nAlgoAccepts.insert(valType(aName, 0));
                    m_nAlgoRejects.insert(valType(aName, 1));
                }
            }
        }

    }
}

// method called once each job just after ending the event loop
void L1GtTrigReport::endJob()
{

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCout;

    myCout << std::dec << std::endl;
    myCout << "L1T-Report " << "----------       Event Summary       ----------\n";
    myCout << "L1T-Report"
    << " Events total = " << m_totalEvents
    << " passed = " << m_globalAccepts
    << " rejected = " << m_totalEvents - m_nErrors - m_globalAccepts
    << " errors = " << m_nErrors
    << "\n";

    myCout << std::endl;
    myCout << "L1T-Report " << "---------- L1 Trigger Global Summary ----------\n\n";

    switch (m_printVerbosity) {
        case 0: {

                myCout
                << std::right << std::setw(20) << "Trigger Menu Key"
                << std::right << std::setw(30) << "Algorithm Key" << " "
                << std::right << std::setw(5)  << "Bit #" << " "
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << "\n";

                std::string dummyMenuKey = "dummyKey";

                for (std::map<int, std::string>::const_iterator
                        it = m_algoBitToName.begin(); it != m_algoBitToName.end(); ++it) {

                    std::string dummyAlgoKey =  it->second;

                    myCout
                    << std::right << std::setw(20) << dummyMenuKey
                    << std::right << std::setw(30) << dummyAlgoKey << " "
                    << std::right << std::setw(5)  << it->first << " "
                    << std::right << std::setw(10) << m_nAlgoAccepts[it->second] << " "
                    << std::right << std::setw(10) << m_nAlgoRejects[it->second] << " "
                    << "\n";

                }

            }

            break;
        case 1: {

                myCout
                << std::right << std::setw(20) << "Trigger Menu Key"
                << std::right << std::setw(30) << "Algorithm Key" << " "
                << std::right << std::setw(5)  << "Bit #" << " "
                << std::right << std::setw(10) << "Prescale" << " "
                << std::right << std::setw(5)  << "Mask" << " "
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << "\n";

                std::string dummyMenuKey = "dummyKey";

                for (std::map<int, std::string>::const_iterator
                        it = m_algoBitToName.begin(); it != m_algoBitToName.end(); ++it) {

                    std::string dummyAlgoKey =  it->second;

                    myCout
                    << std::right << std::setw(20) << dummyMenuKey
                    << std::right << std::setw(30) << dummyAlgoKey << " "
                    << std::right << std::setw(5)  << it->first << " "
                    << std::right << std::setw(10) << m_prescaleFactor[it->second] << " "
                    << std::right << std::setw(5)  << m_triggerMask[it->second] << " "
                    << std::right << std::setw(10) << m_nAlgoAccepts[it->second] << " "
                    << std::right << std::setw(10) << m_nAlgoRejects[it->second] << " "
                    << "\n";

                }

            }

            break;
        default: {
                myCout << "\n\nL1GtTrigReport: Error - no print verbosity level = "
                << m_printVerbosity 
                << " defined! \nCheck available values in the cfi file."
                << "\n";
            }

            break;
    }


    // TODO for other verbosity levels
    // print the trigger menu, the prescale factors and the trigger mask, etc



    myCout << std::endl;
    myCout << "L1T-Report end!" << std::endl;
    myCout << std::endl;

    switch (m_printOutput) {
        case 0: {

                std::cout
                << myCout.str()
                << std::endl;

            }

            break;
        case 1: {

                LogTrace("L1GtTrigReport")
                << myCout.str()
                << std::endl;

            }
            break;

        case 2: {

                edm::LogVerbatim("L1GtTrigReport")
                << myCout.str()
                << std::endl;

            }

            break;
        default: {
                std::cout << "\n\nL1GtTrigReport: Error - no print output = "
                << m_printOutput
                << " defined! \nCheck available values in the cfi file."
                << "\n"
                << std::endl;

            }
            break;
    }



}

// TODO FIXME temporary solution, until L1 trigger menu is implemented as event setup
// get map from algorithm names to algorithm bits
void L1GtTrigReport::getAlgoMap(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{


    // get object maps (one object map per algorithm)
    // the record can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_l1GtObjectMapTag.label(), gtObjectMapRecord);

    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
        gtObjectMapRecord->gtObjectMap();

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
            itMap != objMapVec.end(); ++itMap) {

        int algoBit = (*itMap).algoBitNumber();
        std::string algoNameStr = (*itMap).algoName();

        m_algoBitToName[algoBit] = algoNameStr;

    }

}


