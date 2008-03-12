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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
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

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"

// constructor(s)
L1GtTrigReport::L1GtTrigReport(const edm::ParameterSet& pSet) {

    // boolean flag to select the input record
    // if true, it will use L1GlobalTriggerRecord 
    m_useL1GlobalTriggerRecord = 
        pSet.getParameter<bool>("UseL1GlobalTriggerRecord");

    /// input tag for GT record (L1 GT DAQ record or L1 GT "lite" record):
    m_l1GtRecordInputTag = pSet.getParameter<edm::InputTag>("L1GtRecordInputTag");

    // print verbosity
    m_printVerbosity = pSet.getUntrackedParameter<int>("PrintVerbosity", 0);

    // print output
    m_printOutput = pSet.getUntrackedParameter<int>("PrintOutput", 0);

    LogDebug("L1GtTrigReport") 
        << "\nUse L1GlobalTriggerRecord:   "
        << "\n   (if false: L1GtTrigReport uses L1GlobalTriggerReadoutRecord.)"
        << "\nInput tag for L1 GT record:  "
        << m_l1GtRecordInputTag.label() << " \n" 
        << "\nPrint verbosity level:           " << m_printVerbosity << " \n" 
        << "\nPrint output:                    " << m_printOutput << " \n" 
        << std::endl;

    // initialize cached IDs
    
    //
    m_l1GtStableParCacheID = 0ULL;

    m_numberPhysTriggers = 0;
    m_numberTechnicalTriggers = 0;
    m_numberDaqPartitions = 0;
    m_numberDaqPartitionsMax = 0;
    
    //
    m_l1GtPfAlgoCacheID = 0ULL;
    m_l1GtPfTechCacheID = 0ULL;
    
    m_l1GtTmAlgoCacheID = 0ULL;
    m_l1GtTmTechCacheID = 0ULL;
    
    m_l1GtTmVetoAlgoCacheID = 0ULL;
    m_l1GtTmVetoTechCacheID = 0ULL;
    
    //
    m_l1GtMenuCacheID = 0ULL;
    
    
    // initialize global counters

    // number of events processed
    m_totalEvents = 0;

    //
    m_entryList.clear();
    
    // set the index of physics DAQ partition TODO input parameter?
    m_physicsDaqPartition = 0;

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

    // get / update the stable parameters from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID = 
        evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {
        
        edm::ESHandle< L1GtStableParameters > l1GtStablePar;
        evSetup.get< L1GtStableParametersRcd >().get( l1GtStablePar );        
        m_l1GtStablePar = l1GtStablePar.product();
        
        // number of physics triggers
        m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of technical triggers
        m_numberTechnicalTriggers = m_l1GtStablePar->gtNumberTechnicalTriggers();
        
        // number of DAQ partitions
        m_numberDaqPartitions = 8; // FIXME add it to stable parameters

        if (m_numberDaqPartitionsMax < m_numberDaqPartitions) {
            
            int numberDaqPartitionsOld = m_numberDaqPartitionsMax; 
            m_numberDaqPartitionsMax = m_numberDaqPartitions;
        
            m_globalNrErrors.reserve(m_numberDaqPartitionsMax);
            m_globalNrAccepts.reserve(m_numberDaqPartitionsMax);

            for (unsigned int iDaq = numberDaqPartitionsOld; 
                iDaq < m_numberDaqPartitionsMax; ++iDaq) {

                m_globalNrErrors.push_back(0);
                m_globalNrAccepts.push_back(0);
                                
            }
            
        }

        //
        m_l1GtStableParCacheID = l1GtStableParCacheID;

    }

    // get / update the prescale factors from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtPfAlgoCacheID = 
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {
        
        edm::ESHandle< L1GtPrescaleFactors > l1GtPfAlgo;
        evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd >().get( l1GtPfAlgo );        
        m_l1GtPfAlgo = l1GtPfAlgo.product();
        
        m_prescaleFactorsAlgoTrig = m_l1GtPfAlgo->gtPrescaleFactors();
        
        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }

    unsigned long long l1GtPfTechCacheID = 
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

    if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {
        
        edm::ESHandle< L1GtPrescaleFactors > l1GtPfTech;
        evSetup.get< L1GtPrescaleFactorsTechTrigRcd >().get( l1GtPfTech );        
        m_l1GtPfTech = l1GtPfTech.product();
        
        m_prescaleFactorsTechTrig = m_l1GtPfTech->gtPrescaleFactors();
        
        m_l1GtPfTechCacheID = l1GtPfTechCacheID;

    }

    
    // get / update the trigger mask from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtTmAlgoCacheID = 
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmAlgo;
        evSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( l1GtTmAlgo );        
        m_l1GtTmAlgo = l1GtTmAlgo.product();
        
        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();
        
        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }
    

    unsigned long long l1GtTmTechCacheID = 
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmTech;
        evSetup.get< L1GtTriggerMaskTechTrigRcd >().get( l1GtTmTech );        
        m_l1GtTmTech = l1GtTmTech.product();
        
        m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();
        
        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }
    
    unsigned long long l1GtTmVetoAlgoCacheID = 
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoAlgo;
        evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd >().get( l1GtTmVetoAlgo );        
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();
        
        m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();
        
        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }
    

    unsigned long long l1GtTmVetoTechCacheID = 
        evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoTechCacheID != l1GtTmVetoTechCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoTech;
        evSetup.get< L1GtTriggerMaskVetoTechTrigRcd >().get( l1GtTmVetoTech );        
        m_l1GtTmVetoTech = l1GtTmVetoTech.product();
        
        m_triggerMaskVetoTechTrig = m_l1GtTmVetoTech->gtTriggerMask();
        
        m_l1GtTmVetoTechCacheID = l1GtTmVetoTechCacheID;

    }
    
    // get / update the trigger menu from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtMenuCacheID = evSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
        
        edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
        evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;
        m_l1GtMenu =  l1GtMenu.product();
        
        m_l1GtMenuCacheID = l1GtMenuCacheID;
        
    }

    const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();
    const std::string& menuName = m_l1GtMenu->gtTriggerMenuName();
    
    // ... end EventSetup
    
    // get L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
    // in L1GlobalTriggerRecord, only the physics partition is available
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    edm::Handle<L1GlobalTriggerRecord> gtRecord;

    if (m_useL1GlobalTriggerRecord) {
        iEvent.getByLabel(m_l1GtRecordInputTag, gtRecord);       
    } else {
        iEvent.getByLabel(m_l1GtRecordInputTag, gtReadoutRecord);       
    } 

    bool validRecord = false;
    
    DecisionWord gtDecisionWordBeforeMask;
    DecisionWord gtDecisionWordAfterMask;
    
    if (m_useL1GlobalTriggerRecord) {

        if (gtRecord.isValid()) {

            // get Global Trigger decision and the decision word
            bool gtDecision = gtRecord->decision();
            gtDecisionWordBeforeMask = gtRecord->decisionWordBeforeMask();
            gtDecisionWordAfterMask = gtRecord->decisionWord();

            if (gtDecision) {
                m_globalNrAccepts[m_physicsDaqPartition]++;
            }

            validRecord = true;

        }
        else {

            m_globalNrErrors[m_physicsDaqPartition]++;

            LogDebug("L1GtTrigReport")
                    << "L1GlobalTriggerRecord with input tag "
                    << m_l1GtRecordInputTag.label() << " not found.\n\n"
                    << std::endl;

        }

    }
    else {
        if (gtReadoutRecord.isValid()) {

            // get Global Trigger finalOR and the decision word
            boost::uint16_t gtFinalOR = gtReadoutRecord->finalOR();
            gtDecisionWordBeforeMask = gtReadoutRecord->decisionWord();

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                bool gtDecision = static_cast<bool> (gtFinalOR & (1
                        << iDaqPartition));
                if (gtDecision) {
                    m_globalNrAccepts[iDaqPartition]++;
                }

            }

            validRecord = true;
        }
        else {

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {
                m_globalNrErrors[iDaqPartition]++;
            }

            LogDebug("L1GtTrigReport")
                    << "L1GlobalTriggerReadoutRecord with input tag "
                    << m_l1GtRecordInputTag.label() << " not found.\n\n"
                    << std::endl;

        }

    }
    
    if (validRecord) {

        // loop over algorithms and increase the corresponding counters
        for (CItAlgo itAlgo = algorithmMap.begin(); 
            itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = (itAlgo->second)->algoBitNumber();
            
            // the result before applying the trigger masks is available 
            // in both L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
            bool algResultBeforeMask = gtDecisionWordBeforeMask[algBitNumber];

            int prescaleFactor = m_prescaleFactorsAlgoTrig.at(algBitNumber);

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = 
                    (m_triggerMaskAlgoTrig.at(algBitNumber)) & (1 << iDaqPartition);
                
                bool algResultAfterMask = false;
                
                if (m_useL1GlobalTriggerRecord) {
                    if (iDaqPartition == m_physicsDaqPartition) {
                        // result available already for physics DAQ partition 
                        // in lite record
                        algResultAfterMask = gtDecisionWordAfterMask[algBitNumber];                       
                    } else {
                        // apply the masks for other partitions 
                        algResultAfterMask = algResultBeforeMask;
                        
                        if (triggerMask) {
                            algResultAfterMask = false;
                        }                                                                                
                    }
                 } else {                    
                    // apply the masks for L1GlobalTriggerReadoutRecord
                    algResultAfterMask = algResultBeforeMask;
                    
                    if (triggerMask) {
                        algResultAfterMask = false;
                    }                                
                } 
                
                            
                L1GtTrigReportEntry* entryRep =
                    new L1GtTrigReportEntry(menuName, algName, prescaleFactor, 
                            triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryList.begin(); itEntry
                        != m_entryList.end(); itEntry++) {
                    if ((*entryRep) == *(*itEntry)) {
                        iCount++;
                        // increase the corresponding counter in the list entry
                        (*itEntry)->addValidEntry(algResultAfterMask, algResultBeforeMask);
                    }
                }

                if (iCount == 0) {
                    // if entry not in the list, increase the corresponding counter
                    // and push the entry in the list
                    entryRep->addValidEntry(algResultAfterMask, algResultBeforeMask);
                    m_entryList.push_back(entryRep);
                }
                else {
                    delete entryRep;
                }
            }
        }
    }
    else {

        // loop over algorithms and increase the error counters
        for (CItAlgo itAlgo = algorithmMap.begin();
            itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = (itAlgo->second)->algoBitNumber();

            int prescaleFactor = m_prescaleFactorsAlgoTrig.at(algBitNumber);

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = 
                    (m_triggerMaskAlgoTrig.at(algBitNumber)) & (1 << iDaqPartition);

                L1GtTrigReportEntry* entryRep = 
                    new L1GtTrigReportEntry(menuName, algName, prescaleFactor, 
                            triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {
                    
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

}

// method called once each job just after ending the event loop
void L1GtTrigReport::endJob() {

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCout;

    myCout << std::dec << std::endl;
    myCout << "L1T-Report " << "----------       Event Summary       ----------\n";
    myCout << "L1T-Report\n"; 

    myCout 
    << "\n DAQ partition " 
    << "        Total  " << "       Passed  " << "     Rejected  " << "       Errors  "
    << "\n"  << std::endl;

    for (unsigned int iDaqPartition = 0; 
        iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {
        
        int rejectedEvents = 
            m_totalEvents - m_globalNrErrors[iDaqPartition] - 
                m_globalNrAccepts[iDaqPartition];
        
        if (m_useL1GlobalTriggerRecord && (iDaqPartition != m_physicsDaqPartition)) {
            continue;
        } else {
            
            myCout 
            << std::right << std::setw(13) << iDaqPartition << "  " 
            << std::right << std::setw(13) << m_totalEvents << "  "  
            << std::right << std::setw(13) << m_globalNrAccepts[iDaqPartition] << "  "  
            << std::right << std::setw(13) << rejectedEvents << "  " 
            << std::right << std::setw(13) << m_globalNrErrors[iDaqPartition] 
            << std::endl;

        }
        
     } 
    

    switch (m_printVerbosity) {
        case 0: {

            myCout << "\nL1T-Report " 
            << "---------- L1 Trigger Global Summary - DAQ Partition " 
            << m_physicsDaqPartition 
            << "----------\n\n";
            
            myCout 
                << "\n\n Number of events written after applying L1 prescale factors"
                << " and trigger masks\n"
                << " if not explicitely mentioned.\n\n";  

            myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" << " " 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                << std::right << std::setw(10) << "Passed" << " " 
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" 
                << "\n";

            for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {
                
                if ((*itEntry)->gtDaqPartition() == m_physicsDaqPartition) {

                    myCout 
                    << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName() 
                    << " " 
                    << std::right << std::setw(35) << (*itEntry)->gtAlgoName()
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() 
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() 
                    << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsError() 
                    << " " << "\n";
                }

            }

        }

            break;
        case 1: {

            myCout << "\nL1T-Report " 
            << "---------- L1 Trigger Global Summary - DAQ Partition "
            << m_physicsDaqPartition 
            << "----------\n\n";

            myCout 
                << "\n\n Number of events written after applying L1 prescale factors"
                << " and trigger masks\n"
                << " if not explicitely mentioned.\n\n";  

            myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" << " " 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                << std::right << std::setw(10) << "Prescale" << " " 
                << std::right << std::setw(5)  << "Mask" << " " 
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" 
                << "\n";

            for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {

                if ((*itEntry)->gtDaqPartition() == m_physicsDaqPartition) {
                    myCout 
                    << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName()
                    << " " 
                    << std::right << std::setw(35) << (*itEntry)->gtAlgoName()
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtPrescaleFactor() 
                    << "   " 
                    << std::right << std::setw(2) //<< std::setfill('0')  
                    << std::hex << (*itEntry)->gtTriggerMask() //<< std::setfill(' ') 
                    << std::dec << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() 
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() 
                    << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsError()
                    << "\n";
                }
            }

        }

            break;
        case 10: {

            myCout << "\nL1T-Report " 
            << "---------- L1 Trigger Global Summary - DAQ Partition "
            << m_physicsDaqPartition 
            << "----------\n\n";

            myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" << " " 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                << std::right << std::setw(10) << "Prescale" << " " 
                << std::right << std::setw(5)  << "Mask" << " " 
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" << " " << "\n";

            for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {

                if ((*itEntry)->gtDaqPartition() == m_physicsDaqPartition) {
                    myCout 
                    << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName() 
                    << " " 
                    << std::right << std::setw(35) << (*itEntry)->gtAlgoName() 
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtPrescaleFactor() 
                    << "    " 
                    << std::right << std::setw(2) //<< std::setfill('0')  
                    << std::hex << (*itEntry)->gtTriggerMask() //<< std::setfill(' ') 
                    << std::dec << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAcceptBeforeMask() 
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsRejectBeforeMask() 
                    << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsError()
                    << std::right << std::setw(76) << " "
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() 
                    << " " 
                    << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() 
                    << "\n";
                }
            }

        }

            break;
        case 100: {

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                myCout << "\nL1T-Report " 
                << "---------- L1 Trigger Global Summary - DAQ Partition " 
                << iDaqPartition << " " 
                << "----------\n\n";
            
                myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" << " " 
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                << std::right << std::setw(10) << "Passed" << " " 
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" << " " << "\n";

                for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {
                
                    if ((*itEntry)->gtDaqPartition() == 0) {

                        myCout 
                        << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName()
                        << " " 
                        << std::right << std::setw(35) << (*itEntry)->gtAlgoName()
                        << " " 
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() 
                        << " " 
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() 
                        << " "
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsError() 
                        << "\n";
                    }

                }
            }
        }

            break;
        case 101: {

            for (unsigned int iDaqPartition = 0; 
                iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                myCout << "\nL1T-Report " 
                << "---------- L1 Trigger Global Summary - DAQ Partition " 
                << iDaqPartition << " " 
                << "----------\n\n";

                myCout 
                << std::right << std::setw(20) << "Trigger Menu Key" << " "  
                << std::right << std::setw(35) << "Algorithm Key" << " " 
                << std::right << std::setw(10) << "Prescale" << " " 
                << std::right << std::setw(5)  << "Mask" << " " 
                << std::right << std::setw(10) << "Passed" << " "
                << std::right << std::setw(10) << "Rejected" << " "
                << std::right << std::setw(10) << "Error" 
                << "\n";

                for (CItEntry itEntry = m_entryList.begin(); 
                    itEntry != m_entryList.end(); itEntry++) {

                    if ((*itEntry)->gtDaqPartition() == 0) {
                        myCout 
                        << std::right << std::setw(20) << (*itEntry)->gtTriggerMenuName()
                        << " "  
                        << std::right << std::setw(35) << (*itEntry)->gtAlgoName()
                        << " " 
                        << std::right << std::setw(10) << (*itEntry)->gtPrescaleFactor() 
                        << "   " 
                        << std::right << std::setw(2) //<< std::setfill('0')  
                        << std::hex << (*itEntry)->gtTriggerMask() //<< std::setfill(' ') 
                        << std::dec << " " 
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() 
                        << " " 
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() 
                        << " "
                        << std::right << std::setw(10) << (*itEntry)->gtNrEventsError()
                        << "\n";
                    }
                }
            }
        }

            break;
        default: {
            myCout << "\n\nL1GtTrigReport: Error - no print verbosity level = " 
                << m_printVerbosity
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
            std::cout << "\n\nL1GtTrigReport: Error - no print output = " 
                << m_printOutput
                << " defined! \nCheck available values in the cfi file." << "\n" 
                << std::endl;

        }
            break;
    }

}

