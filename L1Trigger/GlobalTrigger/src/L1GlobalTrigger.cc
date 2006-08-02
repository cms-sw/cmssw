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
 * $Date:$
 * $Revision:$
 */


// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

// system include files
#include <memory>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"

//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructors

L1GlobalTrigger::L1GlobalTrigger(const edm::ParameterSet& iConfig) {

    LogDebug ("Trace") << "Entering L1GlobalTriger constructor";
     
    // register products
    produces<L1GlobalTriggerReadoutRecord>();
    
    // set configuration parameters
    if(!m_gtSetup) m_gtSetup = new L1GlobalTriggerSetup(iConfig);
    
    std::string emptyString;
    m_gtSetup->setTriggerMenu(emptyString);
    
//    // create new PSBs
//    edm::LogInfo("L1GlobalTrigger") << "\n Creating GT PSBs" << endl;
//    m_gtPSB = new L1GlobalTriggerPSB(*this);
  
    // create new GTL
    edm::LogInfo("L1GlobalTrigger") << "\n Creating GT GTL" << endl;
    m_gtGTL = new L1GlobalTriggerGTL(*this);

//    // create new FDL
//    edm::LogInfo("L1GlobalTrigger") << "\n Creating GT FDL" << endl;
//    m_gtFDL = new L1GlobalTriggerFDL(*this);
   
    
    
    
}


// destructor
L1GlobalTrigger::~L1GlobalTrigger() {
 
    if(m_gtSetup) delete m_gtSetup;
    m_gtSetup = 0;

//    delete m_gtPSB;
    delete m_gtGTL;
//    delete m_gtFDL;
}


// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    LogDebug ("Trace") << "L1GlobalTriger::produce";

    using namespace edm;

    // process event iEvent
    
    edm::LogInfo("L1GlobalTrigger") << "\n**** L1GlobalTrigger processing ****\n" << std::endl;

//    // * receive GCT data via PSBs
//    if ( m_gtPSB ) { 
//        edm::LogInfo("L1GlobalTrigger") << "\nL1GlobalTrigger : running PSB\n" << std::endl;
//        m_gtPSB->receiveData(iEvent);
//    }  

    // * receive GMT data via GTL
    if ( m_gtGTL ) {
        edm::LogInfo("L1GlobalTrigger") << "\nL1GlobalTrigger : receive GMT data\n" << std::endl;
        m_gtGTL->receiveData(iEvent); 
    }

    // * run GTL
    if ( m_gtGTL ) {
        edm::LogInfo("L1GlobalTrigger") << "\nL1GlobalTrigger : running GTL\n" << std::endl;
        m_gtGTL->run();
    } 

//    // * run FDL
//    if ( m_gtFDL ) {
//        edm::LogInfo("L1GlobalTrigger") << "\nL1GlobalTrigger : running FDL\n" << std::endl; 
//        m_gtFDL->run();
//    }
    
    // produce the output 
    std::auto_ptr<L1GlobalTriggerReadoutRecord> L1GlobalTriggerReadoutRecord( 
        new L1GlobalTriggerReadoutRecord );
    iEvent.put( L1GlobalTriggerReadoutRecord );

}

// static data members

L1GlobalTriggerSetup* L1GlobalTrigger::m_gtSetup = 0;
