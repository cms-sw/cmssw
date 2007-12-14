/**
 * \class L1GtAnalyzer
 * 
 * 
 * Description: see header file.  
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

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"



// constructor(s)
L1GtAnalyzer::L1GtAnalyzer(const edm::ParameterSet& parSet)
{

    // input tag for GT DAQ record
    m_daqGtInputTag = parSet.getUntrackedParameter<edm::InputTag>(
                          "DaqGtInputTag", edm::InputTag("L1GtEmulDigis"));

    LogDebug("L1GtAnalyzer")
    << "\nInput tag for GT DAQ record: "
    << m_daqGtInputTag.label() << " \n"
    << std::endl;

    // input tag for GT object map collection
    m_gtObjectMapTag = parSet.getUntrackedParameter<edm::InputTag>(
                           "GtObjectMapTag", edm::InputTag("L1GtEmulDigis"));

    LogDebug("L1GtAnalyzer")
    << "\nInput tag for GT object map collection: "
    << m_gtObjectMapTag.label() << " \n"
    << std::endl;

    // input tag for muon collection from GMT
    m_muGmtInputTag = parSet.getUntrackedParameter<edm::InputTag>(
                          "GmtInputTag", edm::InputTag("L1GmtEmulDigis"));

    LogDebug("L1GtAnalyzer")
    << "\nInput tag for muon collection from GMT: "
    << m_muGmtInputTag.label() << " \n"
    << std::endl;

    /// an algorithm and a condition in that algorithm to test the object maps
    m_algoName = parSet.getParameter<std::string>("AlgorithmName");
    m_condName = parSet.getParameter<std::string>("ConditionName");

    LogDebug("L1GtAnalyzer")
    << "\nObject map example for algorithm: " << m_algoName
    << " and condition" << m_condName << " \n"
    << std::endl;

}

// destructor
L1GtAnalyzer::~L1GtAnalyzer()
{}

// member functions

// analyze: decision and decision word
//   bunch cross in event BxInEvent = 0 - L1Accept event
void L1GtAnalyzer::analyzeDecision(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeDecision decision and decision word ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    // get Global Trigger decision and the decision word
    bool gtDecision = gtReadoutRecord->decision();
    DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    // print Global Trigger decision and the decision word
    edm::LogVerbatim("L1GtAnalyzer")
    << "\n GlobalTrigger decision: " << gtDecision << std::endl;

    // print via supplied "print" function (for debug level)
    if ( edm::isDebugEnabled() )
        gtReadoutRecord->printGtDecision(myCoutStream);

    LogDebug("L1GtAnalyzer")
    << myCoutStream.str()
    << std::endl;


    // print technical trigger word via supplied "print" function
    if ( edm::isDebugEnabled() )
        gtReadoutRecord->printTechnicalTrigger(myCoutStream);

    LogDebug("L1GtAnalyzer")
    << myCoutStream.str()
    << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();


    // acces bit
    const unsigned int numberTriggerBits = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    edm::LogVerbatim("L1GtAnalyzer") << std::endl;
    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {
        edm::LogVerbatim("L1GtAnalyzer")
        << "Bit " << iBit <<  ": triger bit = " << gtDecisionWord[iBit] << std::endl;
    }

}

// analyze: test setting decision
//   bunch cross in event BxInEvent = 0 - L1Accept event
void L1GtAnalyzer::analyzeSetDecision(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeSetDecision test setting decision ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get the old record
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    // get Global Trigger decision
    bool gtDecision = gtReadoutRecord->decision();

    // print Global Trigger decision
    edm::LogVerbatim("L1GtAnalyzer")
    << "\n Initial GlobalTrigger decision: " << gtDecision << std::endl;

    // get a new record
    L1GlobalTriggerReadoutRecord* gtReadoutRecordNew = new L1GlobalTriggerReadoutRecord();

    edm::LogVerbatim("L1GtAnalyzer")
    << "\n New record created, with BxInEvent = "
    << gtReadoutRecordNew->gtFdlWord().bxInEvent()
    << std::endl;

    // set a new global decision for bunch cross with L1Accept
    bool newGlobalDecision = true;
    edm::LogVerbatim("L1GtAnalyzer")
    << "\n GlobalTrigger decision to be set to : " << newGlobalDecision
    << std::endl;

    gtReadoutRecordNew->setDecision(newGlobalDecision);

    // get Global Trigger decision
    gtDecision = gtReadoutRecordNew->decision();

    // print new Global Trigger decision
    edm::LogVerbatim("L1GtAnalyzer")
    << "\n GlobalTrigger decision set to : " << gtDecision << "\n\n"
    << std::endl;

    delete gtReadoutRecordNew;

    myCoutStream.str("");
    myCoutStream.clear();

} // end snippet to test setting decision

// print/access L1 objects in bunch cross with L1A
void L1GtAnalyzer::analyzeL1Objects(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeL1Objects print/access L1 objects in L1GlobalTriggerReadoutRecord ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    // print L1 objects in bunch cross with L1A
    gtReadoutRecord->printL1Objects(myCoutStream);

    edm::LogVerbatim("L1GtAnalyzer")
    << myCoutStream.str()
    << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

}


// test muon part in L1GlobalTriggerReadoutRecord
void L1GtAnalyzer::analyzeMuons(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeMuons test muon collection in L1GlobalTriggerReadoutRecord ****\n"
    << std::endl;

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    // get reference to muon collection
    const edm::RefProd<L1MuGMTReadoutCollection>
    muCollRefProd = gtReadoutRecord->muCollectionRefProd();

    if (muCollRefProd.isNull()) {
        edm::LogInfo("L1GtAnalyzer")
        << "Null reference for L1MuGMTReadoutCollection"
        << std::endl;

    } else {
        edm::LogInfo("L1GtAnalyzer")
        << "RefProd address = " << &muCollRefProd
        << std::endl;

        // test all three variants to get muon index 0 in BXInEvent = 0
        unsigned int indexCand = 0;
        int bxInEvent = 0;

        // test first if the record has the required number of candidates
        if ((*muCollRefProd).getRecord(bxInEvent).getGMTCands().size() > indexCand) {
            edm::LogInfo("L1GtAnalyzer")
            << "Three variants to get muon index 0 in BXInEvent = 0"
            << "\n via RefProd, muonCand(indexCand, bxInEvent), muonCand(indexCand)"
            << std::endl;

            L1MuGMTExtendedCand mu00 =
                (*muCollRefProd).getRecord(bxInEvent).getGMTCands().at(indexCand);
            mu00.print();

            L1MuGMTExtendedCand mu00A = gtReadoutRecord->muonCand(indexCand, bxInEvent);
            mu00A.print();

            L1MuGMTExtendedCand mu00B = gtReadoutRecord->muonCand(indexCand);
            mu00B.print();

        }

        // test methods to get GMT records
        std::vector<L1MuGMTReadoutRecord> muRecords = (*muCollRefProd).getRecords();
        edm::LogInfo("L1GtAnalyzer")
        << "\nNumber of records in the GMT RefProd readout collection = "
        << muRecords.size()
        << std::endl;

        for (std::vector<L1MuGMTReadoutRecord>::const_iterator itMu = muRecords.begin();
                itMu < muRecords.end(); ++itMu) {

            std::vector<L1MuGMTExtendedCand>
            exc = itMu->getGMTCands();
            for(std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter = exc.begin();
                    gmt_iter != exc.end(); gmt_iter++) {

                (*gmt_iter).print();
            }

        }

        // test GMT record for BxInEvent = 0  (default argument)
        std::vector<L1MuGMTExtendedCand> muRecord0 = gtReadoutRecord->muonCands();
        edm::LogInfo("L1GtAnalyzer")
        << "\nRecord for BxInEvent = 0 using default argument"
        << std::endl;

        for(std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter = muRecord0.begin();
                gmt_iter != muRecord0.end(); gmt_iter++) {
            (*gmt_iter).print();
        }

        // test GMT record for BxInEvent = 1
        std::vector<L1MuGMTExtendedCand> muRecord1 = gtReadoutRecord->muonCands(1);
        edm::LogInfo("L1GtAnalyzer")
        << "\nRecord for BxInEvent = 1 using BxInEvent argument"
        << std::endl;

        for(std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter = muRecord1.begin();
                gmt_iter != muRecord1.end(); gmt_iter++) {
            (*gmt_iter).print();
        }
    }


}

// analyze: object map record
void L1GtAnalyzer::analyzeObjectMap(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeObjectMap object map record ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get a handle to the object map record
    // the record can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_gtObjectMapTag.label(), gtObjectMapRecord);

    // get all object maps
    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
        gtObjectMapRecord->gtObjectMap();

    // print every object map via the implemented print
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator it = objMapVec.begin();
            it != objMapVec.end(); ++it) {

        (*it).print(myCoutStream);
    }

    //
    const CombinationsInCond* comb =
        gtObjectMapRecord->getCombinationsInCond(m_algoName, m_condName);

    // number of combinations
    if (comb != 0) {
        myCoutStream << "\n  Number of combinations passing ("
        << m_algoName << ", " << m_condName << "): "
        << comb->size()
        << std::endl;
    }

    // condition result
    const bool result = gtObjectMapRecord->getConditionResult(m_algoName, m_condName);

    myCoutStream << "\n  Result for condition " << m_condName
    << " in algorithm " << m_algoName << ": " << result
    << std::endl;

    // print all the stuff if at LogDebug level
    LogDebug("L1GtAnalyzer")
    << "Test gtObjectMapRecord in L1GlobalTrigger \n\n" << myCoutStream.str() << "\n\n"
    << std::endl;
    myCoutStream.str("");
    myCoutStream.clear();

}


// analyze each event: event loop
void L1GtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    LogDebug("L1GtAnalyzer") << "\n**** L1GtAnalyzer::analyze starting ****\n"
    << std::endl;

    // analyze: decision and decision word
    //   bunch cross in event BxInEvent = 0 - L1Accept event
    analyzeDecision(iEvent, iSetup);

    // analyze: test setting decision
    //   bunch cross in event BxInEvent = 0 - L1Accept event
    analyzeSetDecision(iEvent, iSetup);

    // print/access L1 objects in bunch cross with L1A
    //    analyzeL1Objects(iEvent, iSetup); // TODO FIXME temporary disable it

    // test muon part in L1GlobalTriggerReadoutRecord
    analyzeMuons(iEvent, iSetup);

    // analyze: object map record
    analyzeObjectMap(iEvent, iSetup);

}


// method called once each job just before starting event loop
void L1GtAnalyzer::beginJob(const edm::EventSetup&)
{

    // empty

}

// method called once each job just after ending the event loop
void L1GtAnalyzer::endJob()
{

    // empty

}

