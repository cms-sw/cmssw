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
L1GtAnalyzer::L1GtAnalyzer(const edm::ParameterSet& iConfig)
{
    m_pSet = &iConfig;

    // input tag for GT DAQ record
    m_daqGtInputTag = m_pSet->getUntrackedParameter<edm::InputTag>(
                          "GtInputTag", edm::InputTag("L1GtEmul"));

    LogDebug("L1GtAnalyzer")
    << "\nInput tag for GT DAQ record: "
    << m_daqGtInputTag.label() << " \n"
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

            std::vector<L1MuGMTExtendedCand> exc = itMu->getGMTCands();
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
    iEvent.getByLabel(m_daqGtInputTag.label(), gtObjectMapRecord);

    // get all object maps
    const std::vector<L1GlobalTriggerObjectMap>&
    objMapVec = gtObjectMapRecord->gtObjectMap();

    // print every object map via the implemented print
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator it = objMapVec.begin();
            it != objMapVec.end(); ++it) {

        (*it).print(myCoutStream);
    }

    // TODO read them from cfg file?
    std::string algoNameVal = "Egamma_and_Muon";
    std::string condNameVal = "Muon_7";

    //    std::string algoNameVal = "Algo_ComplexSyntax";
    //    std::string condNameVal = "Mu_60";
    //
    const CombinationsInCond*
    comb = gtObjectMapRecord->getCombinationsInCond(algoNameVal, condNameVal);

    // number of combinations
    if (comb != 0) {
        myCoutStream << "\n  Number of combinations passing ("
        << algoNameVal << ", " << condNameVal << "): "
        << comb->size()
        << std::endl;
    }

    // condition result
    const bool result = gtObjectMapRecord->getConditionResult(algoNameVal, condNameVal);

    myCoutStream << "\n  Result for condition " << condNameVal
    << " in algorithm " << algoNameVal << ": " << result
    << std::endl;

    // print all the stuff if at LogDebug level
    LogDebug("L1GtAnalyzer")
    << "Test gtObjectMapRecord in L1GlobalTrigger \n\n" << myCoutStream.str() << "\n\n"
    << std::endl;
    myCoutStream.str("");
    myCoutStream.clear();

}

// analyze: seed muon HLT using the object map record
void L1GtAnalyzer::analyzeObjectMapMuons(
    const edm::Event& iEvent,
    const edm::EventSetup& iSetup)
{

    LogDebug("L1GtAnalyzer")
    << "\n**** L1GtAnalyzer::analyzeObjectMapMuons seed muon HLT using the object map record ****\n"
    << std::endl;

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCoutStream;

    // get a handle to the object map record
    // the record can come only from emulator - no hardware ObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtObjectMapRecord);

    // algorithm and condition names to be read from cfg file?
    // after the trigger menu will be implemented as event setup, it should be
    // possible to search the menu for all algorithms containing muon conditions
    // and for the requirements imposed in each condition

    std::string algoNameVal = "Egamma_and_Muon";
    std::string condNameVal = "Muon_7";

    //    std::string algoNameVal = "Algo_ComplexSyntax";
    //    std::string condNameVal = "Mu_60";
    //

    // condition result
    const bool result = gtObjectMapRecord->getConditionResult(algoNameVal, condNameVal);

    myCoutStream << "\n  Result for condition " << condNameVal
    << " in algorithm " << algoNameVal << ": " << result
    << std::endl;

    // getCombinationsInCond returns all the combinations passing the requirements
    // imposed in condition condNameVal from algorithm with name algoNameVal
    // if the result is false/zero, no combinations are returned

    // TODO ask HLT people about "NOT condition" case

    if (result) {

        const CombinationsInCond*
        comb = gtObjectMapRecord->getCombinationsInCond(algoNameVal, condNameVal);

        // number of combinations must be tested if one not tests the condition
        // result, the pointer can have zero value!
        if (comb != 0) {
            myCoutStream << "\n  Number of combinations passing ("
            << algoNameVal << ", " << condNameVal << "): "
            << comb->size()
            << std::endl;
        } else {
            myCoutStream << "\n  No combinations passing ("
            << algoNameVal << ", " << condNameVal << ")"
            << std::endl;

            // print all the stuff if at LogDebug level
            LogDebug("L1GtAnalyzer")
            << "\nTest muon seeds coming from ("
            << algoNameVal << ", " << condNameVal << ")\n"
            << myCoutStream.str() << "\n\n"
            << std::endl;
            myCoutStream.str("");
            myCoutStream.clear();

            return;
        }

        // get data from Global Muon Trigger record

        // TODO FIXME get the InputTag from provenence of object map record
        edm::InputTag muGmtInputTag("gmt");

        myCoutStream
        << "\n  L1GtAnalyzer receiving muon data from GMT record with input tag "
        << muGmtInputTag.label()
        << std::endl;

        edm::Handle<L1MuGMTReadoutCollection> muonData;
        iEvent.getByLabel(muGmtInputTag.label(), muonData);

        // object maps are saved for iBxInEvent = 0
        int iBxInEvent = 0;

        myCoutStream
        << "\n  Muon seeds from ("
        << algoNameVal << ", " << condNameVal << "):"
        << std::endl;

        // loop over combinations

        CombinationsInCond::const_iterator itVV;
        for(itVV  = comb->begin(); itVV != comb->end(); itVV++) {

            // loop over objects in every single combinations

            myCoutStream << "\n   Combination: ( ";
            std::copy((*itVV).begin(), (*itVV).end(),
                      std::ostream_iterator<int> (myCoutStream, " "));
            myCoutStream << "); " << std::endl;


            SingleCombInCond::const_iterator itV;
            for (itV = (*itVV).begin(); itV != (*itVV).end(); itV++) {

                int indexCand = (*itV);
                L1MuGMTExtendedCand muSeed = (*muonData).getRecord(iBxInEvent).getGMTCands()[indexCand];

                // do whatever you want with the seeds
                // here, print transverse momentum, charge, eta, phi, ...

                myCoutStream << "    Muon index:   " << indexCand << std::endl;
                myCoutStream << "      pT =        " << muSeed.ptValue() << " GeV" << std::endl;
                myCoutStream << "      charge =    " << muSeed.charge() << std::endl;
                myCoutStream << "      eta =       " << muSeed.etaValue() << std::endl;
                myCoutStream << "      phi =       " << muSeed.phiValue() << std::endl;
                myCoutStream << "      quality =   " << muSeed.quality() << std::endl;
                myCoutStream << "      detector =  " << muSeed.detector() << std::endl;
                myCoutStream << "      rank =      " << muSeed.rank() << std::endl;
                myCoutStream << "      isolation = " << muSeed.isol() << std::endl;
                myCoutStream << "      mip =       " << muSeed.mip() << std::endl;

            }

        }

    }

    // print all the stuff if at LogDebug level
    LogDebug("L1GtAnalyzer")
    << "\nTest muon seeds coming from ("
    << algoNameVal << ", " << condNameVal << ")\n"
    << myCoutStream.str() << "\n\n"
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

    // object map available only in emulator
    if (m_daqGtInputTag.label() == "L1GtEmul") {

        // analyze: object map record
        analyzeObjectMap(iEvent, iSetup);

        // analyze: seed muon HLT using the object map record
        analyzeObjectMapMuons(iEvent, iSetup);

    }


}


// method called once each job just before starting event loop
void L1GtAnalyzer::beginJob(const edm::EventSetup&)
{}

// method called once each job just after ending the event loop
void L1GtAnalyzer::endJob()
{}

// static data members
const edm::ParameterSet* L1GtAnalyzer::m_pSet = 0;

