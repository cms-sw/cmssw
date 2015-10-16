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
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"

// system include files
#include <memory>

#include <iostream>
#include <iomanip>

#include<map>
#include<set>
#include <cmath>
#include <string>

#include "boost/lexical_cast.hpp"

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
L1GtTrigReport::L1GtTrigReport(const edm::ParameterSet& pSet) :

    // initialize cached IDs

    //
    m_l1GtStableParCacheID( 0ULL ),

    m_numberPhysTriggers( 0 ),
    m_numberTechnicalTriggers( 0 ),
    m_numberDaqPartitions( 0 ),
    m_numberDaqPartitionsMax( 0 ),

    //
    m_l1GtPfAlgoCacheID( 0ULL ),
    m_l1GtPfTechCacheID( 0ULL ),

    m_l1GtTmAlgoCacheID( 0ULL ),
    m_l1GtTmTechCacheID( 0ULL ),

    m_l1GtTmVetoAlgoCacheID( 0ULL ),
    m_l1GtTmVetoTechCacheID( 0ULL ),

    //
    m_l1GtMenuCacheID( 0ULL ),

    // boolean flag to select the input record
    // if true, it will use L1GlobalTriggerRecord
    m_useL1GlobalTriggerRecord( pSet.getParameter<bool>("UseL1GlobalTriggerRecord") ),

    /// input tag for GT record (L1 GT DAQ record or L1 GT "lite" record):
    m_l1GtRecordInputTag( pSet.getParameter<edm::InputTag>("L1GtRecordInputTag") ),
    m_l1GtRecordInputToken1( m_useL1GlobalTriggerRecord
        ? consumes<L1GlobalTriggerRecord>(m_l1GtRecordInputTag)
        : edm::EDGetTokenT<L1GlobalTriggerRecord>() ),
    m_l1GtRecordInputToken2( not m_useL1GlobalTriggerRecord
        ? consumes<L1GlobalTriggerReadoutRecord>(m_l1GtRecordInputTag)
        : edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>() ),

    // print verbosity
    m_printVerbosity( pSet.getUntrackedParameter<int>("PrintVerbosity", 2) ),

    // print output
    m_printOutput( pSet.getUntrackedParameter<int>("PrintOutput", 3) ),

    // initialize global counters

    // number of events processed
    m_totalEvents( 0 ),

    //
    m_entryList(),
    m_entryListTechTrig(),

    // set the index of physics DAQ partition TODO input parameter?
    m_physicsDaqPartition( 0 )

{
    LogDebug("L1GtTrigReport") << "\n  Use L1GlobalTriggerRecord:   " << m_useL1GlobalTriggerRecord
            << "\n   (if false: L1GtTrigReport uses L1GlobalTriggerReadoutRecord.)"
            << "\n  Input tag for L1 GT record:  " << m_l1GtRecordInputTag.label() << " \n"
            << "\n  Print verbosity level:           " << m_printVerbosity << " \n"
            << "\n  Print output:                    " << m_printOutput << " \n" << std::endl;

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

    for (ItEntry itEntry = m_entryListTechTrig.begin(); itEntry != m_entryListTechTrig.end(); itEntry++) {
        if (*itEntry != 0) {
            delete *itEntry;
            *itEntry = 0;
        }
    }

    m_entryListTechTrig.clear();

}

// member functions


// method called once each job just before starting event loop
void L1GtTrigReport::beginJob() {

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

        edm::ESHandle<L1GtStableParameters> l1GtStablePar;
        evSetup.get<L1GtStableParametersRcd>().get(l1GtStablePar);
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

            for (unsigned int iDaq = numberDaqPartitionsOld; iDaq < m_numberDaqPartitionsMax; ++iDaq) {

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

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);
        m_l1GtPfAlgo = l1GtPfAlgo.product();

        m_prescaleFactorsAlgoTrig = & ( m_l1GtPfAlgo->gtPrescaleFactors() );

        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }

    unsigned long long l1GtPfTechCacheID =
            evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

    if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfTech;
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().get(l1GtPfTech);
        m_l1GtPfTech = l1GtPfTech.product();

        m_prescaleFactorsTechTrig = & ( m_l1GtPfTech->gtPrescaleFactors() );

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

        m_l1GtMenuCacheID = l1GtMenuCacheID;

        LogDebug("L1GtTrigReport") << "\n  Changing L1 menu to : \n"
                << m_l1GtMenu->gtTriggerMenuName() << "\n\n" << std::endl;

    }


    const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();
    const AlgorithmMap& technicalTriggerMap = m_l1GtMenu->gtTechnicalTriggerMap();

    const std::string& menuName = m_l1GtMenu->gtTriggerMenuName();

    // ... end EventSetup

    // get L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
    // in L1GlobalTriggerRecord, only the physics partition is available
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    edm::Handle<L1GlobalTriggerRecord> gtRecord;

    if (m_useL1GlobalTriggerRecord) {
        iEvent.getByToken(m_l1GtRecordInputToken1, gtRecord);
    } else {
        iEvent.getByToken(m_l1GtRecordInputToken2, gtReadoutRecord);
    }

    bool validRecord = false;

    unsigned int pfIndexAlgo = 0; // get them later from the record
    unsigned int pfIndexTech = 0;

    DecisionWord gtDecisionWordBeforeMask;
    DecisionWord gtDecisionWordAfterMask;

    TechnicalTriggerWord technicalTriggerWordBeforeMask;
    TechnicalTriggerWord technicalTriggerWordAfterMask;

    if (m_useL1GlobalTriggerRecord) {

        if (gtRecord.isValid()) {

            // get Global Trigger decision and the decision word
            bool gtDecision = gtRecord->decision();

            gtDecisionWordBeforeMask = gtRecord->decisionWordBeforeMask();
            gtDecisionWordAfterMask = gtRecord->decisionWord();

            technicalTriggerWordBeforeMask = gtRecord->technicalTriggerWordBeforeMask();
            technicalTriggerWordAfterMask = gtRecord->technicalTriggerWord();

            if (gtDecision) {
                m_globalNrAccepts[m_physicsDaqPartition]++;
            }

            pfIndexAlgo = gtRecord->gtPrescaleFactorIndexAlgo();
            pfIndexTech = gtRecord->gtPrescaleFactorIndexTech();

            validRecord = true;

        } else {

            m_globalNrErrors[m_physicsDaqPartition]++;

            edm::LogWarning("L1GtTrigReport") << "\n  L1GlobalTriggerRecord with input tag "
                    << m_l1GtRecordInputTag.label() << " not found."
                    << "\n  Event classified as Error\n\n"
                    << std::endl;

        }

    } else {
        if (gtReadoutRecord.isValid()) {

            // check if the readout record has size greater than zero, then proceeds
            const std::vector<L1GtFdlWord>& fdlVec = gtReadoutRecord->gtFdlVector();
            size_t fdlVecSize = fdlVec.size();

            if (fdlVecSize > 0) {

                LogDebug("L1GtTrigReport") << "\n  L1GlobalTriggerReadoutRecord with input tag "
                        << m_l1GtRecordInputTag.label() << " has gtFdlVector of size " << fdlVecSize
                        << std::endl;

                // get Global Trigger finalOR and the decision word
                boost::uint16_t gtFinalOR = gtReadoutRecord->finalOR();

                gtDecisionWordBeforeMask = gtReadoutRecord->decisionWord();
                technicalTriggerWordBeforeMask = gtReadoutRecord->technicalTriggerWord();

                for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                    bool gtDecision = static_cast<bool>(gtFinalOR & ( 1 << iDaqPartition ));
                    if (gtDecision) {
                        m_globalNrAccepts[iDaqPartition]++;
                    }

                }

                pfIndexAlgo
                        = static_cast<unsigned int>( ( gtReadoutRecord->gtFdlWord() ).gtPrescaleFactorIndexAlgo());
                pfIndexTech
                        = static_cast<unsigned int>( ( gtReadoutRecord->gtFdlWord() ).gtPrescaleFactorIndexTech());

                validRecord = true;

            } else {

                for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {
                    m_globalNrErrors[iDaqPartition]++;
                }

                edm::LogWarning("L1GtTrigReport") << "\n  L1GlobalTriggerReadoutRecord with input tag "
                        << m_l1GtRecordInputTag.label() << " has gtFdlVector of size " << fdlVecSize
                        << "\n  Invalid L1GlobalTriggerReadoutRecord!"
                        << "\n  Event classified as Error\n\n"
                        << std::endl;

                validRecord = false;

            }

        } else {

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {
                m_globalNrErrors[iDaqPartition]++;
            }

            edm::LogWarning("L1GtTrigReport") << "\n  L1GlobalTriggerReadoutRecord with input tag "
                    << m_l1GtRecordInputTag.label() << " not found."
                    << "\n  Event classified as Error\n\n"
                    << std::endl;

        }

    }

    // get the prescale factor set used in the actual luminosity segment
    const std::vector<int>& prescaleFactorsAlgoTrig =
            ( *m_prescaleFactorsAlgoTrig ).at(pfIndexAlgo);

    const std::vector<int>& prescaleFactorsTechTrig =
            ( *m_prescaleFactorsTechTrig ).at(pfIndexTech);


    if (validRecord) {

        // loop over algorithms and increase the corresponding counters
        for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = ( itAlgo->second ).algoBitNumber();

            // the result before applying the trigger masks is available
            // in both L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
            bool algResultBeforeMask = gtDecisionWordBeforeMask[algBitNumber];

            int prescaleFactor = prescaleFactorsAlgoTrig.at(algBitNumber);

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = ( m_triggerMaskAlgoTrig.at(algBitNumber) ) & ( 1
                        << iDaqPartition );

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

                L1GtTrigReportEntry* entryRep = new L1GtTrigReportEntry(
                        menuName, algName, prescaleFactor, triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {
                    if ( ( *entryRep ) == * ( *itEntry )) {
                        iCount++;
                        // increase the corresponding counter in the list entry
                        ( *itEntry )->addValidEntry(algResultAfterMask, algResultBeforeMask);
                    }
                }

                if (iCount == 0) {
                    // if entry not in the list, increase the corresponding counter
                    // and push the entry in the list
                    entryRep->addValidEntry(algResultAfterMask, algResultBeforeMask);
                    m_entryList.push_back(entryRep);
                } else {
                    delete entryRep;
                }
            }
        }

        // loop over technical triggers and increase the corresponding counters
        for (CItAlgo itAlgo = technicalTriggerMap.begin(); itAlgo != technicalTriggerMap.end(); itAlgo++) {
        //for (unsigned int iTechTrig = 0; iTechTrig < m_numberTechnicalTriggers; ++iTechTrig) {

            std::string ttName = itAlgo->first;
            int ttBitNumber = ( itAlgo->second ).algoBitNumber();
            // std::string ttName = boost::lexical_cast<std::string>(iTechTrig);
            // int ttBitNumber = iTechTrig;

            // the result before applying the trigger masks is available
            // in both L1GlobalTriggerReadoutRecord or L1GlobalTriggerRecord
            bool ttResultBeforeMask = technicalTriggerWordBeforeMask[ttBitNumber];

            int prescaleFactor = prescaleFactorsTechTrig.at(ttBitNumber);

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = ( m_triggerMaskTechTrig.at(ttBitNumber) ) & ( 1
                        << iDaqPartition );

                bool ttResultAfterMask = false;

                if (m_useL1GlobalTriggerRecord) {
                    if (iDaqPartition == m_physicsDaqPartition) {
                        // result available already for physics DAQ partition
                        // in lite record
                        ttResultAfterMask = technicalTriggerWordAfterMask[ttBitNumber];
                    } else {
                        // apply the masks for other partitions
                        ttResultAfterMask = ttResultBeforeMask;

                        if (triggerMask) {
                            ttResultAfterMask = false;
                        }
                    }
                } else {
                    // apply the masks for L1GlobalTriggerReadoutRecord
                    ttResultAfterMask = ttResultBeforeMask;

                    if (triggerMask) {
                        ttResultAfterMask = false;
                    }
                }

                L1GtTrigReportEntry* entryRep = new L1GtTrigReportEntry(
                        menuName, ttName, prescaleFactor, triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry
                        != m_entryListTechTrig.end(); itEntry++) {
                    if ( ( *entryRep ) == * ( *itEntry )) {
                        iCount++;
                        // increase the corresponding counter in the list entry
                        ( *itEntry )->addValidEntry(ttResultAfterMask, ttResultBeforeMask);
                    }
                }

                if (iCount == 0) {
                    // if entry not in the list, increase the corresponding counter
                    // and push the entry in the list
                    entryRep->addValidEntry(ttResultAfterMask, ttResultBeforeMask);
                    m_entryListTechTrig.push_back(entryRep);
                } else {
                    delete entryRep;
                }
            }
        }

    } else {

        // loop over algorithms and increase the error counters
        for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

            std::string algName = itAlgo->first;
            int algBitNumber = ( itAlgo->second ).algoBitNumber();

            int prescaleFactor = prescaleFactorsAlgoTrig.at(algBitNumber);

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = ( m_triggerMaskAlgoTrig.at(algBitNumber) ) & ( 1
                        << iDaqPartition );

                L1GtTrigReportEntry* entryRep = new L1GtTrigReportEntry(
                        menuName, algName, prescaleFactor, triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( ( *entryRep ) == * ( *itEntry )) {
                        iCount++;
                        // increase the corresponding counter in the list entry
                        ( *itEntry )->addErrorEntry();
                    }
                }

                if (iCount == 0) {
                    // if entry not in the list, increase the corresponding counter
                    // and push the entry in the list
                    entryRep->addErrorEntry();
                    m_entryList.push_back(entryRep);
                } else {
                    delete entryRep;
                }
            }

        }

        // loop over technical triggers and increase the error counters
        // FIXME move to names when technical triggers available in menu
        //for (CItAlgo itAlgo = technicalTriggerMap.begin(); itAlgo != technicalTriggerMap.end(); itAlgo++) {
        for (unsigned int iTechTrig = 0; iTechTrig < m_numberTechnicalTriggers; ++iTechTrig) {

            //std::string ttName = itAlgo->first;
            //int ttBitNumber = ( itAlgo->second ).algoBitNumber();
            std::string ttName = boost::lexical_cast<std::string>(iTechTrig);
            int ttBitNumber = iTechTrig;

            int prescaleFactor = prescaleFactorsTechTrig.at(ttBitNumber);

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                unsigned int triggerMask = ( m_triggerMaskTechTrig.at(ttBitNumber) ) & ( 1
                        << iDaqPartition );

                L1GtTrigReportEntry* entryRep = new L1GtTrigReportEntry(
                        menuName, ttName, prescaleFactor, triggerMask, iDaqPartition);

                int iCount = 0;

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry
                        != m_entryListTechTrig.end(); itEntry++) {

                    if ( ( *entryRep ) == * ( *itEntry )) {
                        iCount++;
                        // increase the corresponding counter in the list entry
                        ( *itEntry )->addErrorEntry();
                    }
                }

                if (iCount == 0) {
                    // if entry not in the list, increase the corresponding counter
                    // and push the entry in the list
                    entryRep->addErrorEntry();
                    m_entryListTechTrig.push_back(entryRep);
                } else {
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
    myCout << "L1T-Report " << "Total number of events processed: " << m_totalEvents << "\n";
    myCout << "L1T-Report\n";

    myCout
        << "\n"
        << "   DAQ partition "
        << "           Total "
        << " Passed[finalOR] "
        << "        Rejected "
        << "          Errors "
        << "\n" << std::endl;

    for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

        int rejectedEvents = m_totalEvents - m_globalNrErrors[iDaqPartition]
                - m_globalNrAccepts[iDaqPartition];

        if (m_useL1GlobalTriggerRecord && ( iDaqPartition != m_physicsDaqPartition )) {
            continue;
        } else {

            myCout
                << std::right << std::setw(16) << iDaqPartition << " "
                << std::right << std::setw(16) << m_totalEvents << " "
                << std::right << std::setw(16) << m_globalNrAccepts[iDaqPartition] << " "
                << std::right << std::setw(16) << rejectedEvents << " "
                << std::right << std::setw(16) << m_globalNrErrors[iDaqPartition] << std::endl;

        }

    }

    // get the list of menus for the sample analyzed
    //
    std::set<std::string> menuList;
    typedef std::set<std::string>::const_iterator CItL1Menu;

    for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {
        menuList.insert( ( *itEntry )->gtTriggerMenuName());
    }

    myCout
            << "\nThe following L1 menus were used for this sample: " << std::endl;
    for (CItL1Menu itMenu = menuList.begin(); itMenu != menuList.end(); itMenu++) {
        myCout << "  " << ( *itMenu ) << std::endl;
    }
    myCout << "\n" << std::endl;

    switch (m_printVerbosity) {
        case 0: {

            myCout
                << "\nL1T-Report " << "---------- L1 Trigger Global Summary - DAQ Partition "
                << m_physicsDaqPartition << "----------\n\n";

            myCout
                << "\n\n Number of events written after applying L1 prescale factors"
                << " and trigger masks\n" << " if not explicitly mentioned.\n\n";

            for (CItL1Menu itMenu = menuList.begin(); itMenu != menuList.end(); itMenu++) {

                myCout << "\nReport for L1 menu:  " << (*itMenu) << "\n"
                        << std::endl;

                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error"
                    << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry
                        != m_entryList.end(); itEntry++) {

                    if (((*itEntry)->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        myCout
                            << std::right << std::setw(45) << (*itEntry)->gtAlgoName() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsError()
                            << "\n";
                    }

                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error"
                    << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry
                        != m_entryListTechTrig.end(); itEntry++) {

                    if (((*itEntry)->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        myCout
                            << std::right << std::setw(45) << (*itEntry)->gtAlgoName() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << (*itEntry)->gtNrEventsError()
                            << "\n";
                    }

                }
            }

        }

            break;
        case 1: {

            myCout << "\nL1T-Report " << "---------- L1 Trigger Global Summary - DAQ Partition "
                    << m_physicsDaqPartition << "----------\n\n";

            myCout << "\n\n Number of events written after applying L1 prescale factors"
                    << " and trigger masks\n" << " if not explicitly mentioned.\n\n";

            for (CItL1Menu itMenu = menuList.begin(); itMenu != menuList.end(); itMenu++) {

                myCout << "\nReport for L1 menu:  " << (*itMenu) << "\n"
                        << std::endl;
                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Prescale" << " "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( (( *itEntry )->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << "    "
                            << std::right << std::setw(2) //<< std::setfill('0')
                            << std::hex << ( *itEntry )->gtTriggerMask() //<< std::setfill(' ')
                            << std::dec << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << "\n";
                    }
                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Prescale" << " "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry != m_entryListTechTrig.end(); itEntry++) {

                    if ( (( *itEntry )->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << "    "
                            << std::right << std::setw(2) //<< std::setfill('0')
                            << std::hex << ( *itEntry )->gtTriggerMask() //<< std::setfill(' ')
                            << std::dec << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << "\n";
                    }
                }
            }

        }

            break;
        case 2: {


            for (CItL1Menu itMenu = menuList.begin(); itMenu != menuList.end(); itMenu++) {

                myCout << "\nReport for L1 menu:  " << ( *itMenu ) << "\n"
                        << std::endl;

                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( ( ( *itEntry )->gtDaqPartition() == m_physicsDaqPartition )
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        int nrEventsAccept = ( *itEntry )->gtNrEventsAccept();
                        int nrEventsReject = ( *itEntry )->gtNrEventsReject();
                        int nrEventsError = ( *itEntry )->gtNrEventsError();

                        myCout
                            << std::right << std::setw(45) << (( *itEntry )->gtAlgoName()) << " "
                            << std::right << std::setw(10) << nrEventsAccept << " "
                            << std::right << std::setw(10) << nrEventsReject << " "
                            << std::right << std::setw(10) << nrEventsError << "\n";

                    }
                }

                // efficiency and its statistical error

                myCout << "\n\n"
                    << std::right << std::setw(45) << "Algorithm Key" << "    "
                    << std::right << std::setw(10) << "Efficiency " << " "
                    << std::right << std::setw(10) << "Stat error (%)" << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( ( ( *itEntry )->gtDaqPartition() == 0 )
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        int nrEventsAccept = ( *itEntry )->gtNrEventsAccept();
                        int nrEventsReject = ( *itEntry )->gtNrEventsReject();
                        int nrEventsError = ( *itEntry )->gtNrEventsError();

                        int totalEvents = nrEventsAccept + nrEventsReject + nrEventsError;

                        // efficiency and their statistical error
                        float eff = 0.;
                        float statErrEff = 0.;

                        if (totalEvents != 0) {
                            eff = static_cast<float> (nrEventsAccept)
                                    / static_cast<float> (totalEvents);
                            statErrEff = sqrt(eff * ( 1.0 - eff )
                                    / static_cast<float> (totalEvents));

                        }

                        myCout
                            << std::right << std::setw(45) << (( *itEntry )->gtAlgoName()) << " "
                            << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                            << 100.*eff << " +- "
                            << std::right << std::setw(10) << std::setprecision(2)
                            << 100.*statErrEff << "\n";


                    }

                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry
                        != m_entryListTechTrig.end(); itEntry++) {

                    if ( ( ( *itEntry )->gtDaqPartition() == m_physicsDaqPartition )
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        int nrEventsAccept = ( *itEntry )->gtNrEventsAccept();
                        int nrEventsReject = ( *itEntry )->gtNrEventsReject();
                        int nrEventsError = ( *itEntry )->gtNrEventsError();

                        myCout
                            << std::right << std::setw(45) << (( *itEntry )->gtAlgoName()) << " "
                            << std::right << std::setw(10) << nrEventsAccept << " "
                            << std::right << std::setw(10) << nrEventsReject << " "
                            << std::right << std::setw(10) << nrEventsError << "\n";

                    }
                }

                // efficiency and its statistical error

                myCout << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << "    "
                    << std::right << std::setw(10) << "Efficiency " << " "
                    << std::right << std::setw(10) << "Stat error (%)" << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry
                        != m_entryListTechTrig.end(); itEntry++) {

                    if ( ( ( *itEntry )->gtDaqPartition() == 0 )
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {

                        int nrEventsAccept = ( *itEntry )->gtNrEventsAccept();
                        int nrEventsReject = ( *itEntry )->gtNrEventsReject();
                        int nrEventsError = ( *itEntry )->gtNrEventsError();

                        int totalEvents = nrEventsAccept + nrEventsReject + nrEventsError;

                        // efficiency and their statistical error
                        float eff = 0.;
                        float statErrEff = 0.;

                        if (totalEvents != 0) {
                            eff = static_cast<float> (nrEventsAccept)
                                    / static_cast<float> (totalEvents);
                            statErrEff = sqrt(eff * ( 1.0 - eff )
                                    / static_cast<float> (totalEvents));

                        }

                        myCout
                            << std::right << std::setw(45) << (( *itEntry )->gtAlgoName()) << " "
                            << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                            << 100.*eff << " +- "
                            << std::right << std::setw(10) << std::setprecision(2)
                            << 100.*statErrEff << "\n";


                    }

                }

            }


        }
            break;

        case 10: {

            myCout << "\nL1T-Report " << "---------- L1 Trigger Global Summary - DAQ Partition "
                    << m_physicsDaqPartition << "----------\n\n";

            for (CItL1Menu itMenu = menuList.begin(); itMenu != menuList.end(); itMenu++) {

                myCout << "\nReport for L1 menu:  " << ( *itMenu ) << "\n"
                        << std::endl;
                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Prescale" << "  "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(25) << "Before Mask" << " "
                    << std::right << std::setw(30) << "After Mask" << " "
                    << std::right << std::setw(22) << "Error"
                    << "\n"
                    << std::right << std::setw(64) << " " << std::setw(15) << "Passed"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Rejected"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Passed"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Rejected"
                    << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( (( *itEntry )->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << " "
                            << std::right << std::setw(5)  << " " << std::hex << ( *itEntry )->gtTriggerMask() << std::dec << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsAcceptBeforeMask() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsRejectBeforeMask() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsError()
                            << "\n";
                    }
                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Prescale" << "  "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(25) << "Before Mask" << " "
                    << std::right << std::setw(30) << "After Mask" << " "
                    << std::right << std::setw(22) << "Error"
                    << "\n"
                    << std::right << std::setw(64) << " " << std::setw(15) << "Passed"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Rejected"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Passed"
                    << std::right << std::setw(1)  << " " << std::setw(15) << "Rejected"
                    << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry != m_entryListTechTrig.end(); itEntry++) {

                    if ( (( *itEntry )->gtDaqPartition() == m_physicsDaqPartition)
                            && ( ( *itEntry )->gtTriggerMenuName() == *itMenu )) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << " "
                            << std::right << std::setw(5)  << " " << std::hex << ( *itEntry )->gtTriggerMask() << std::dec << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsAcceptBeforeMask() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsRejectBeforeMask() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(15) << ( *itEntry )->gtNrEventsError()
                            << "\n";
                    }
                }
            }
        }

            break;
        case 100: {

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                myCout << "\nL1T-Report "
                        << "---------- L1 Trigger Global Summary - DAQ Partition " << iDaqPartition
                        << " " << "----------\n\n";

                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( (( *itEntry )->gtDaqPartition() == 0)) {

                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << "\n";
                    }

                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry != m_entryListTechTrig.end(); itEntry++) {

                    if ( ( *itEntry )->gtDaqPartition() == 0) {

                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << std::right << std::setw(20) << ( *itEntry )->gtTriggerMenuName()
                            << "\n";
                    }

                }

            }
        }

            break;
        case 101: {

            for (unsigned int iDaqPartition = 0; iDaqPartition < m_numberDaqPartitions; ++iDaqPartition) {

                myCout << "\nL1T-Report "
                        << "---------- L1 Trigger Global Summary - DAQ Partition " << iDaqPartition
                        << " " << "----------\n\n";

                myCout
                    << std::right << std::setw(45) << "Algorithm Key" << " "
                    << std::right << std::setw(10) << "Prescale" << " "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryList.begin(); itEntry != m_entryList.end(); itEntry++) {

                    if ( ( *itEntry )->gtDaqPartition() == 0) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << "   "
                            << std::right << std::setw(2) //<< std::setfill('0')
                            << std::hex << ( *itEntry )->gtTriggerMask() //<< std::setfill(' ')
                            << std::dec << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << "\n";
                    }
                }

                myCout
                    << "\n\n"
                    << std::right << std::setw(45) << "Technical Trigger Key" << " "
                    << std::right << std::setw(10) << "Prescale" << " "
                    << std::right << std::setw(5)  << "Mask" << " "
                    << std::right << std::setw(10) << "Passed" << " "
                    << std::right << std::setw(10) << "Rejected" << " "
                    << std::right << std::setw(10) << "Error" << std::setw(2) << " "
                    << "\n";

                for (CItEntry itEntry = m_entryListTechTrig.begin(); itEntry != m_entryListTechTrig.end(); itEntry++) {

                    if ( ( *itEntry )->gtDaqPartition() == 0) {
                        myCout
                            << std::right << std::setw(45) << ( *itEntry )->gtAlgoName() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtPrescaleFactor() << "   "
                            << std::right << std::setw(2) //<< std::setfill('0')
                            << std::hex << ( *itEntry )->gtTriggerMask() //<< std::setfill(' ')
                            << std::dec << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsAccept() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsReject() << " "
                            << std::right << std::setw(10) << ( *itEntry )->gtNrEventsError() << std::setw(2) << " "
                            << "\n";
                    }
                }

            }
        }

            break;
        default: {
            myCout
                << "\n\nL1GtTrigReport: Error - no print verbosity level = " << m_printVerbosity
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
        case 3: {

            edm::LogInfo("L1GtTrigReport") << myCout.str();

        }

            break;
        default: {
            std::cout
                << "\n\n  L1GtTrigReport: Error - no print output = " << m_printOutput
                << " defined! \n  Check available values in the cfi file." << "\n" << std::endl;

        }
            break;
    }

}

