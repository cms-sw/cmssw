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
#include <string>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"

// constructor(s)
L1GtTrigReport::L1GtTrigReport(const edm::ParameterSet& pSet) {

    // input tag for GT DAQ record
    m_l1GtDaqInputTag = pSet.getParameter<edm::InputTag>("L1GtDaqInputTag");

    // print verbosity
    m_printVerbosity = pSet.getUntrackedParameter<int>("PrintVerbosity", 0);

    // print output
    m_printOutput = pSet.getUntrackedParameter<int>("PrintOutput", 0);

    LogDebug("L1GtTrigReport") 
        << "\nInput tag for L1 GT DAQ record:  "
        << m_l1GtDaqInputTag.label() << " \n" 
        << "\nPrint verbosity level:           " << m_printVerbosity << " \n" 
        << "\nPrint output:                    " << m_printOutput << " \n" 
        << std::endl;

    // initialize global counters

    // number of events processed
    m_totalEvents = 0;

    // number of events with error (EDProduct[s] not found)
    m_globalNrErrors = 0;

    // number of events accepted by any of the L1 algorithm
    m_globalNrAccepts = 0;

    //
    m_entryList.clear();
}

// destructor
L1GtTrigReport::~L1GtTrigReport() {

    for (ItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {
        if (*itEntry != 0) {
            delete *itEntry;
            *itEntry = 0;
        }
    }

    m_entryList.clear();
}


// member functions


// method called once each job just before starting event loop
void L1GtTrigReport::beginJob(const edm::EventSetup& evSetup) {

    // empty

}

// analyze each event
void L1GtTrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

    // increase the number of processed events
    m_totalEvents++;

    // get EventSetup
    //     prescale factos and trigger mask
    edm::ESHandle< L1GtPrescaleFactors> l1GtPfAlgo;
    evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo) ;

    edm::ESHandle< L1GtTriggerMask> l1GtTmAlgo;
    evSetup.get< L1GtTriggerMaskAlgoTrigRcd>().get(l1GtTmAlgo) ;

    //     the trigger menu from the EventSetup

    edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
    evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;

    const AlgorithmMap& algorithmMap = l1GtMenu->gtAlgorithmMap();
    const std::string& menuName = l1GtMenu->gtTriggerMenuName();

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_l1GtDaqInputTag, gtReadoutRecord);

    if (gtReadoutRecord.isValid()) {

        // get Global Trigger decision and the decision word
        bool gtDecision = gtReadoutRecord->decision();
        DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

        if (gtDecision) {
            m_globalNrAccepts++;
        }

        // loop over algorithms and increase the corresponding counters
        for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = (itAlgo->second)->algoBitNumber();
            bool algResult = gtDecisionWord[algBitNumber];

            int prescaleFactor = l1GtPfAlgo->gtPrescaleFactors().at(algBitNumber);
            unsigned int triggerMask = l1GtTmAlgo->gtTriggerMask().at(algBitNumber);

            L1GtTrigReportEntry* entryRep = 
                new L1GtTrigReportEntry(menuName, algName, prescaleFactor, triggerMask);

            int iCount = 0;

            for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {
                if ((*entryRep) == *(*itEntry)) {
                    iCount++;
                    // increase the corresponding counter in the list entry
                    (*itEntry)->addValidEntry(algResult);
                }
            }

            if (iCount == 0) {
                // if entry not in the list, increase the corresponding counter
                // and push the entry in the list
                entryRep->addValidEntry(algResult);
                m_entryList.push_back(entryRep);
            }
            else {
                delete entryRep;
            }
        }
    }
    else {

        m_globalNrErrors++;

        LogDebug("L1GtTrigReport") << "L1GlobalTriggerReadoutRecord with input tag "
            << m_l1GtDaqInputTag.label() << " not found.\n\n" << std::endl;

        // loop over algorithms and increase the error counters
        for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = (itAlgo->second)->algoBitNumber();

            int prescaleFactor = l1GtPfAlgo->gtPrescaleFactors().at(algBitNumber);
            unsigned int triggerMask = l1GtTmAlgo->gtTriggerMask().at(algBitNumber);

            L1GtTrigReportEntry* entryRep = 
                new L1GtTrigReportEntry(menuName, algName, prescaleFactor, triggerMask);

            int iCount = 0;

            for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {
                if ((*entryRep) == *(*itEntry)) {
                    iCount++;
                    // increase the corresponding counter in the list entry
                    (*itEntry)->addErrorEntry();
                }
            }

            if (iCount == 0) {
                // if entry not in the list, increase the corresponding counter
                // and push the entry in the list
                entryRep->addErrorEntry();
                m_entryList.push_back(entryRep);
            }
            else {
                delete entryRep;
            }

        }
    }

}

// method called once each job just after ending the event loop
void L1GtTrigReport::endJob() {

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCout;

    myCout << std::dec << std::endl;
    myCout << "L1T-Report " << "----------       Event Summary       ----------\n";
    myCout << "L1T-Report\n" 
    << "\n    Events: total =    " << m_totalEvents 
    << "\n            passed =   " << m_globalNrAccepts 
    << "\n            rejected = " << m_totalEvents - m_globalNrErrors - m_globalNrAccepts
    << "\n            errors =   " << m_globalNrErrors << "\n" 
    << std::endl;
    
    myCout << "L1T-Report " << "---------- L1 Trigger Global Summary ----------\n\n";

    switch (m_printVerbosity) {
        case 0: {

            myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                //<< std::right << std::setw(5)  << "Bit #" << " " 
                << std::right << std::setw(10) << "Passed" << " " 
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" << " " << "\n";

            for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                myCout 
                    << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName() 
                    << std::right << std::setw(35) << (*itEntry)->gtAlgoName() << " " 
                    //<< std::right << std::setw(5)  << algBitNumber << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsError() << " " << "\n";

            }

        }

            break;
        case 1: {

            myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                //<< std::right << std::setw(5)  << "Bit #" << " " 
                << std::right << std::setw(10) << "Prescale" << " " 
                << std::right << std::setw(5)  << "Mask" << " " 
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" << " " << "\n";

            for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                myCout 
                    << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName() 
                    << std::right << std::setw(35) << (*itEntry)->gtAlgoName() << " " 
                    //<< std::right << std::setw(5)  << algBitNumber << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtPrescaleFactor() << "   " 
                    << std::right << std::setw(2) << std::setfill('0')  
                    << std::hex << (*itEntry)->gtTriggerMask() << std::setfill(' ') << std::dec << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsError() << " " << "\n";

            }

        }

            break;
        default: {
            myCout << "\n\nL1GtTrigReport: Error - no print verbosity level = " << m_printVerbosity
                << " defined! \nCheck available values in the cfi file." << "\n";
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

            std::cout << myCout.str() << std::endl;

        }

            break;
        case 1: {

            LogTrace("L1GtTrigReport") << myCout.str() << std::endl;

        }
            break;

        case 2: {

            edm::LogVerbatim("L1GtTrigReport") << myCout.str() << std::endl;

        }

            break;
        default: {
            std::cout << "\n\nL1GtTrigReport: Error - no print output = " << m_printOutput
                << " defined! \nCheck available values in the cfi file." << "\n" << std::endl;

        }
            break;
    }

}

