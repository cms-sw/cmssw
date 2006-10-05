/**
 * \class L1GlobalTrigger 
 * 
 * 
 * 
 * Description: L1 Global Trigger
 * Implementation: see L1GlobalTrigger.h
 *    
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 */


// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

// system include files
#include <memory>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructors

L1GlobalTrigger::L1GlobalTrigger(const edm::ParameterSet& iConfig) {

    LogDebug ("Trace") << "Entering L1GlobalTriger constructor";
     
    // register products
    produces<L1GlobalTriggerReadoutRecord>();
    
    // set configuration parameters
    if(!m_gtSetup) m_gtSetup = new L1GlobalTriggerSetup(*this, iConfig);
    
    std::string emptyString;
    m_gtSetup->setTriggerMenu(emptyString);
    
    // create new PSBs
    LogDebug("L1GlobalTrigger") << "\n Creating GT PSBs" << std::endl;
    m_gtPSB = new L1GlobalTriggerPSB(*this);
  
    // create new GTL
    LogDebug("L1GlobalTrigger") << "\n Creating GT GTL" << std::endl;
    m_gtGTL = new L1GlobalTriggerGTL(*this);

    // create new FDL
    LogDebug("L1GlobalTrigger") << "\n Creating GT FDL" << std::endl;
    m_gtFDL = new L1GlobalTriggerFDL(*this);
       
}

// destructor
L1GlobalTrigger::~L1GlobalTrigger() {
 
    if(m_gtSetup) delete m_gtSetup;
    m_gtSetup = 0;

    delete m_gtPSB;
    delete m_gtGTL;
    delete m_gtFDL;
}

// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    using namespace edm;

    // process event iEvent
    
    LogDebug("L1GlobalTrigger") << "\n**** L1GlobalTrigger::produce ****\n" << std::endl;

    // * receive GCT data via PSBs
    if ( m_gtPSB ) { 
        LogDebug("L1GlobalTrigger") << "\nL1GlobalTrigger : running PSB\n" << std::endl;
        m_gtPSB->receiveData(iEvent);
    }  

    // * receive GMT data via GTL
    if ( m_gtGTL ) {
        LogDebug("L1GlobalTrigger") << "\nL1GlobalTrigger : receiving GMT data\n" << std::endl;
        m_gtGTL->receiveData(iEvent); 
    }

    // * run GTL
    if ( m_gtGTL ) {
        LogDebug("L1GlobalTrigger") << "\nL1GlobalTrigger : running GTL\n" << std::endl;
        m_gtGTL->run();
    } 

    // * run FDL
    if ( m_gtFDL ) {
        LogDebug("L1GlobalTrigger") << "\nL1GlobalTrigger : running FDL\n" << std::endl; 
        m_gtFDL->run();
        
        LogDebug("L1GlobalTrigger") 
            << "\nDecision word:\n" << m_gtFDL->getDecisionWord()
            << "\nDecision:\n" << m_gtFDL->getDecision()
            << std::endl;
    }
    
    // * produce the L1GlobalTriggerReadoutRecord
    LogDebug("L1GlobalTrigger") 
        << "\nL1GlobalTrigger : producing L1GlobalTriggerReadoutRecord\n"
        << std::endl; 
        
    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerReadoutRecord );

    // ** set the decision word and decision in L1GlobalTriggerReadoutRecord

    // [ ... convert decision word from std::bitset to std::vector<bool>
    //       TODO remove this block when changing DecisionWord to std::bitset    

    const unsigned int numberTriggerBits = L1GlobalTriggerReadoutRecord::NumberPhysTriggers;
    
    std::bitset<numberTriggerBits> fdlDecision = m_gtFDL->getDecisionWord();
    L1GlobalTriggerReadoutRecord::DecisionWord fdlDecisionVec(numberTriggerBits);    

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {
        
        bool bitValue = fdlDecision.test( iBit );        
        fdlDecisionVec[ iBit ] = bitValue;   
    }
    
    gtReadoutRecord->setDecisionWord(fdlDecisionVec);

    // ... ]

//    gtReadoutRecord->setDecisionWord(m_gtFDL->getDecisionWord());        
    gtReadoutRecord->setDecision(m_gtFDL->getDecision());

    if ( edm::isDebugEnabled() ) gtReadoutRecord->print();

    // **             
    iEvent.put( gtReadoutRecord );

}

// static data members

L1GlobalTriggerSetup* L1GlobalTrigger::m_gtSetup = 0;
