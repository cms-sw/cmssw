/**
 * \class L1GtAnalyzer
 * 
 * 
 * 
 * Description: see header file 
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

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



// constructor(s)
L1GtAnalyzer::L1GtAnalyzer(const edm::ParameterSet& iConfig) {
   // initialization

}

// destructor
L1GtAnalyzer::~L1GtAnalyzer() {
 
}

// member functions

// analyze each event: event loop 
void L1GtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

    LogDebug("L1GtAnalyzer") << "\n**** L1GtAnalyzer::analyze starting ****\n" 
        << std::endl;

    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel("L1GtEmul", gtReadoutRecord);

    // get Global Trigger decision and the decision word
    bool gtDecision = gtReadoutRecord->decision();
    L1GlobalTriggerReadoutSetup::DecisionWord gtDecisionWord = gtReadoutRecord->decisionWord();

    // print Global Trigger decision and the decision word
    edm::LogVerbatim("L1GtAnalyzer") 
        << "\n GlobalTrigger decision: " << gtDecision << std::endl;
    
    // print via supplied "print" function (for debug level) 
    std::ostringstream myCoutStream;
    if ( edm::isDebugEnabled() ) gtReadoutRecord->printGtDecision(myCoutStream);
        
    LogDebug("L1GtAnalyzer") 
        << myCoutStream.str()
        << std::endl;
    

    // print technical trigger word via supplied "print" function 
    myCoutStream.str(""); myCoutStream.clear();    
    if ( edm::isDebugEnabled() ) gtReadoutRecord->printTechnicalTrigger(myCoutStream);
    
    LogDebug("L1GtAnalyzer") 
        << myCoutStream.str()
        << std::endl;
    

    // acces bit 
    const unsigned int numberTriggerBits = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;
    
    edm::LogVerbatim("L1GtAnalyzer") << std::endl;        
    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {        
        edm::LogVerbatim("L1GtAnalyzer") 
            << "Bit " << iBit <<  ": triger bit = " << gtDecisionWord[iBit] << std::endl;        
    }
    
    // print L1 objects in bunch cross with L1A
    myCoutStream.str(""); myCoutStream.clear();    
    gtReadoutRecord->printL1Objects(myCoutStream);
    
    edm::LogVerbatim("L1GtAnalyzer") 
        << myCoutStream.str()
        << std::endl;

     // test muon part in L1GlobalTriggerReadoutRecord
    
    edm::LogInfo("L1GtAnalyzer")
        << "\n Test muon collection in the GT readout record"
        << std::endl;

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
        unsigned int bxInEvent = 0;

        edm::LogInfo("L1GtAnalyzer")
            << "Three variants to get muon index 0 in BXInEvent = 0"
            << "\n via RefProd, muonCand(indexCand, bxInEvent), muonCand(indexCand)"
            << std::endl;

        L1MuGMTExtendedCand mu00 = (*muCollRefProd).getRecord(bxInEvent).getGMTCands()[indexCand];
        mu00.print();
        
        L1MuGMTExtendedCand mu00A = gtReadoutRecord->muonCand(indexCand, bxInEvent);
        mu00A.print();
        
        L1MuGMTExtendedCand mu00B = gtReadoutRecord->muonCand(indexCand);
        mu00B.print();
 
        // test methods to get GMT records    
        std::vector<L1MuGMTReadoutRecord> muRecords = (*muCollRefProd).getRecords();
        edm::LogInfo("L1GtAnalyzer")
            << "\nNumber of records in the GMT RefProd readout collection = " 
            << muRecords.size()
            << std::endl;

        for (std::vector<L1MuGMTReadoutRecord>::const_iterator itMu = muRecords.begin(); 
            itMu < muRecords.end(); ++itMu) {
            
            std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter;
            std::vector<L1MuGMTExtendedCand> exc = itMu->getGMTCands();
            for(gmt_iter = exc.begin(); gmt_iter != exc.end(); gmt_iter++) {
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


    // snippet to test setting decision 
    //   bunch cross in event BxInEvent = 0 - L1Accept event    
    L1GlobalTriggerReadoutRecord* gtReadoutRecordNew = new L1GlobalTriggerReadoutRecord();

    edm::LogVerbatim("L1GtAnalyzer") 
        << "\n Snippet to test setting decision functions"
        << "\n New record created, with BxInEvent = " 
        << gtReadoutRecordNew->gtFdlWord().bxInEvent()
        << std::endl;
    
    // get Global Trigger decision
    gtDecision = gtReadoutRecordNew->decision();
    
    // print Global Trigger decision
    edm::LogVerbatim("L1GtAnalyzer") 
        << "\n Initial GlobalTrigger decision: " << gtDecision << std::endl;

    // set a new global decision for bunch cross with L1Accept
    bool newGlobalDecision = true;
    gtReadoutRecordNew->setDecision(newGlobalDecision);
    
    // get Global Trigger decision
    gtDecision = gtReadoutRecordNew->decision();

    // print new Global Trigger decision
    edm::LogVerbatim("L1GtAnalyzer") 
        << "\n GlobalTrigger decision set to : " << gtDecision << std::endl;
    
    delete gtReadoutRecordNew;

    // end snippet to test setting decision
  
}


// method called once each job just before starting event loop
void L1GtAnalyzer::beginJob(const edm::EventSetup&)
{
}

// method called once each job just after ending the event loop
void L1GtAnalyzer::endJob() {

}

// static data members 

