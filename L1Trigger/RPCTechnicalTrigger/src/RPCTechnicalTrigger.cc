// $Id: 

//-----------------------------------------------------------------------------
// Implementation file for class : RPCTechnicalTrigger
//
// 2008-10-15 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

// Include files

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCTechnicalTrigger.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessDigiGlobalSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessDigiLocalSignal.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

//Data Formats
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//Technical trigger bits
#include <DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h>
#include <DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h>

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig) {
  
  //...........................................................................
  m_debugMode = iConfig.getUntrackedParameter<int>("DebugMode", 0);
  m_testFile = iConfig.getUntrackedParameter<std::string>("TestDatafile", std::string("testdata.txt"));
  m_verbosity = iConfig.getUntrackedParameter<int>("Verbosity", 0);
  m_triggerMode = iConfig.getParameter<int>("TriggerMode");
  m_rpcDigiLabel = iConfig.getParameter<edm::InputTag>("RPCDigiLabel");
  m_ttBits = iConfig.getParameter< std::vector<unsigned> >("BitNumbers");
  m_ttNames = iConfig.getParameter< std::vector<std::string> >("BitNames");
  
  if ( m_verbosity ) {
    LogTrace("RPCTechnicalTrigger")
      << m_triggerMode << '\n'
      << m_rpcDigiLabel << '\n'
      << std::endl;
  }
  
  //... Debug mode:
  // 0 - no debug
  // 1 - use ascii file to test emulator
  
  //... Trigger mode:
  // 1 - Global
  // 2 - Local
  // 3 - Local Or Global
  
  if ( m_debugMode != 1 ) m_debugMode = 0;
  
  //...........................................................................
  //... There are three Technical Trigger Units Boards: 1 can handle 2 Wheels
  //... nWheels sets the number of wheels attached to board with index boardIndex
  //...

  m_boardIndex[0] = 1;
  m_boardIndex[1] = 2;
  m_boardIndex[2] = 3;

  m_nWheels[0]    = 2;
  m_nWheels[1]    = 1;
  m_nWheels[2]    = 2;
  
  m_ttu[0] = new TTUEmulator( m_boardIndex[0] , m_nWheels[0] );
  m_ttu[1] = new TTUEmulator( m_boardIndex[1] , m_nWheels[1] );
  m_ttu[2] = new TTUEmulator( m_boardIndex[2] , m_nWheels[2] );
  
  //...........................................................................
  
  m_ievt = 0;
  m_cand = 0;
  m_maxTtuBoards = 3;
  m_maxBits = 5;
  
  produces<L1GtTechnicalTriggerRecord>();
  
}


RPCTechnicalTrigger::~RPCTechnicalTrigger()
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger: object starts deletion" << std::endl;
  
  delete m_ttu[0];
  delete m_ttu[1];
  delete m_ttu[2];
  
  if ( m_debugMode == 1) {
    if (m_signal) delete m_signal;
  }
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger: object deleted" << '\n';
    
}

//=============================================================================
void RPCTechnicalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<RPCDigiCollection> pIn;
  
  std::vector<L1GtTechnicalTrigger> ttVec( m_ttBits.size() );
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  
  iEvent.getByLabel(m_rpcDigiLabel, pIn);
  
  if (m_verbosity && !pIn.isValid()) {
    edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label" << '\n';
    return;
  }
  
  if ( m_debugMode == 1 )
    //... Reading data from an ascii file 
    LogDebug("RPCTechnicalTrigger") << "Reading data from ascii file. Only for testing purposes." << '\n';
  
  else {
    
    switch( m_triggerMode ) {
      
    case 1:
      //... Extracting signal from Digi: Global approach
      LogDebug("RPCTechnicalTrigger") << "Setting up a new signal object: global" << '\n';
      m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessDigiGlobalSignal( m_rpcGeometry, pIn ));
      LogDebug("RPCTechnicalTrigger") << "signal object created" << '\n';
      break;
      
    case 2:
      //... Extracting signal from Digi: Local approach
      LogDebug("RPCTechnicalTrigger") << "Setting up a new signal object: local" << '\n';
      m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessDigiLocalSignal ( m_rpcGeometry, pIn ));
      LogDebug("RPCTechnicalTrigger") << "signal object created" << '\n';
      break;
      
    default:
      edm::LogError("RPCTechnicalTrigger") << "Case not implemented" << '\n';
    }
    
  }
  
  bool status(false);
  status = m_signal->next();
  if ( !status) return;
  
  RPCInputSignal * input = m_signal->retrievedata();
  
  //. distribute data to different TTU emulators and process it
  
  m_triggerbits.reset();
  
  int indx=0;
  
  for(int k=0; k< m_maxTtuBoards; ++k) {
    
    indx=k*2;
    
    if ( m_debugMode == 1 ) {
      // process test full chain using ascii data
      m_ttu[k]->processtest( input );
      m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
      m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
      continue;
    }
    
    switch( m_triggerMode ) {
      
    case 1:
      m_ttu[k]->processglobal( input );
      m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
      m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
      break;
      
    case 2:
      m_ttu[k]->processlocal( input );
      m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
      m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
      break;
      
    default:
      edm::LogError("RPCTechnicalTrigger") << "Case not implemented" << '\n';
    }
    
  }
  
  int bx = 0;
  bool bit = true;
  
  //.. write results to technical trigger bits
  
  for(int i = 0; i < m_maxBits; ++i) {
    ttVec.at(i)=L1GtTechnicalTrigger(m_ttNames.at(i), m_ttBits.at(i), bx, bit) ;
  }
  
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(output);
  
  //... reset data map for next event
  
  input->clear();
  m_triggerbits.reset();
  
  //...
  
  if( m_debugMode == 0 ) delete m_signal;
  
  ++m_ievt;
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
  
}

// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginRun(edm::Run& iRun, const edm::EventSetup& evtSetup)
{
  
  bool status(false);
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::beginJob> starts" << std::endl;
  
  //.   Set up RPC geometry
  
  evtSetup.get<MuonGeometryRecord>().get( m_rpcGeometry );
  
  //..  Get Board Specifications (hardware configuration)
  
  edm::ESHandle<RBCBoardSpecs> pRBCSpecs;
  evtSetup.get<RBCBoardSpecsRcd>().get(pRBCSpecs);
  m_rbcspecs = pRBCSpecs.product();
  
  edm::ESHandle<TTUBoardSpecs> pTTUSpecs;
  evtSetup.get<TTUBoardSpecsRcd>().get(pTTUSpecs);
  
  m_ttuspecs = pTTUSpecs.product();
  
  for (int k=0; k < m_maxTtuBoards; ++k )
    m_ttu[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
  
  //... Initialize all
  
  for (int k=0; k < m_maxTtuBoards; ++k )
    status = m_ttu[k]->initialise();
  
  //..........
  
  if ( m_debugMode == 1 ) {
    //. Read RBC input for an ascii file
    m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessTestSignal( m_testFile.c_str()) );
  }
  
  
}


// ------------ method called once each job just after ending the event loop  ------------

void RPCTechnicalTrigger::endJob() 
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::endJob>" << std::endl;
  
}

void RPCTechnicalTrigger::printinfo()
{
  
  for (int k=0; k < m_maxTtuBoards; ++k )
    m_ttu[k]->printinfo();
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);
