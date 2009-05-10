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
#include "L1Trigger/RPCTechnicalTrigger/src/RPCTechnicalTrigger.h"
#include "L1Trigger/RPCTechnicalTrigger/src/ProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/ProcessDigiGlobalSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/ProcessDigiLocalSignal.h"

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

RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig)
{
  
  //...........................................................................
  
  m_debugmode    = iConfig.getUntrackedParameter<int>("DebugMode"     , 0);
  
  m_triggermode  = iConfig.getUntrackedParameter<int>("TriggerMode"   , 1);
  
  m_testfile     = iConfig.getUntrackedParameter<std::string>("TestDatafile", std::string("testdata.txt"));
  
  m_GMTLabel     = iConfig.getUntrackedParameter<edm::InputTag>("GMTInputTag", edm::InputTag("gmt"));
  
  m_ttBits         = iConfig.getParameter< std::vector<unsigned> >("BitNumbers");
  
  m_ttNames        = iConfig.getParameter< std::vector<std::string> >("BitNames");
  
  //... Debug mode:
  // 0 - no debug
  // 1 - use ascii file to test emulator
  
  //... Trigger mode:
  // 1 - Global
  // 2 - Local
  // 3 - Local Or Global
  
  if ( m_debugmode != 1 ) m_debugmode = 0;
  
  //...........................................................................
  //... There are three Technical Trigger Units: 1 per 2 Wheels
  
  m_ttu[0] = new TTUEmulator( 1 , 2 );
  m_ttu[1] = new TTUEmulator( 2 , 1 );
  m_ttu[2] = new TTUEmulator( 3 , 2 );
  
  //...........................................................................
  
  m_ievt = 0;
  m_cand = 0;
  
  produces<L1GtTechnicalTriggerRecord>();
  
}


RPCTechnicalTrigger::~RPCTechnicalTrigger()
{
  
  std::cout << "RPCTechnicalTrigger: object starts deletion" << std::endl;
  
  delete m_ttu[0];
  delete m_ttu[1];
  delete m_ttu[2];
  
  if ( m_debugmode == 1) {
    if (m_signal) delete m_signal;
  }
  
  std::cout << "RPCTechnicalTrigger: object deleted" << std::endl;
    
}

//=============================================================================
void RPCTechnicalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<RPCDigiCollection> pIn;
  
  std::vector<L1GtTechnicalTrigger> ttVec( m_ttBits.size() );
  
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
 
  try {
    iEvent.getByLabel("muonRPCDigis", pIn);
  }
  catch (...) {
    edm::LogError("analyze") << "can't find RPCDigiCollection with label muonRPCDigis" << '\n';
    return; }
  
  if ( m_debugmode == 1 )
    //... Reading data from an ascii file 
    edm::LogInfo("analyze") << "Reading data from ascii file. Only for testing purposes." << '\n';
  
  else {
    
    switch( m_triggermode ) {
      
    case 1:
      //... Extracting signal from Digi: Global approach
      edm::LogInfo("analyze") << "Setting up a new signal object: global" << '\n';
      m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessDigiGlobalSignal( m_rpcGeometry, pIn ));
      edm::LogInfo("analyze") << "signal object created" << '\n';
      break;
      
    case 2:
      //... Extracting signal from Digi: Local approach
      edm::LogInfo("analyze") << "Setting up a new signal object: local" << '\n';
      m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessDigiLocalSignal ( m_rpcGeometry, pIn ));
      edm::LogInfo("analyze") << "signal object created" << '\n';
      break;
      
    default:
      edm::LogError("analyze") << "Case not implemented" << '\n';
    }
    
  }
  
  bool status(false);
  status = m_signal->next();
  if ( !status) return;
  
  RPCInputSignal * input = m_signal->retrievedata();
  
  //. distribute data to different TTU emulators and process it
  
  m_triggerbits.reset();
  
  int indx=0;
  
  for(int k=0; k<3; ++k) {
    
    indx=k*2;
    
    if ( m_debugmode == 1 ) {
      // process test full chain using ascii data
      m_ttu[k]->processtest( input );
      m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
      m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
      continue;
    }
    
    switch( m_triggermode ) {
      
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
      edm::LogError("analyze") << "Case not implemented" << '\n';
    }
    
  }
  
  //.. analyse results from emulation
  
  m_trigger.reset();
  
  for(int k=0; k<6; ++k) {
    bool tmp = m_triggerbits[k];
    m_trigger[0] = ( m_trigger[0] || tmp );
  }
  
  int bx = 0;
  bool bit = true;
  
  //.. write results to technical trigger bits
  
  for(int i = 0; i < 5; ++i) {
    ttVec.at(i)=L1GtTechnicalTrigger(m_ttNames.at(i), m_ttBits.at(i), bx, bit) ;
  }
  
  output->setGtTechnicalTrigger(ttVec);    
  
  iEvent.put(output);
  
  //... reset data map for next event
  
  input->clear();
  
  //...
  
  if( m_debugmode == 0 ) delete m_signal;
  
  ++m_ievt;
  
  std::cout << "RPCTechnicalTrigger> end of analyze" << std::endl;
  
}

// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginJob(const edm::EventSetup& _evtSetup)
{
  
  bool status(false);
  
  std::cout << "RPCTechnicalTrigger::beginJob> starts" << std::endl;
  
  //.   Set up RPC geometry
  
  _evtSetup.get<MuonGeometryRecord>().get( m_rpcGeometry );
  
  //..  Get Board Specifications (hardware configuration)
  
  edm::ESHandle<RBCBoardSpecs> pRBCSpecs;
  _evtSetup.get<RBCBoardSpecsRcd>().get(pRBCSpecs);
  m_rbcspecs = pRBCSpecs.product();
  
  edm::ESHandle<TTUBoardSpecs> pTTUSpecs;
  _evtSetup.get<TTUBoardSpecsRcd>().get(pTTUSpecs);
  
  m_ttuspecs = pTTUSpecs.product();
  
  for (int k=0; k < 3; ++k )
    m_ttu[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
  
  //... Initialize all
  
  for (int k=0; k < 3; ++k )
    status = m_ttu[k]->initialise();
  
  //..........
  
  if ( m_debugmode == 1 ) {
    //. Read RBC input for an ascii file
    m_signal  = dynamic_cast<ProcessInputSignal*>(new ProcessTestSignal( m_testfile.c_str()) );
  }
  
  
}

// ------------ method called once each job just after ending the event loop  ------------

void RPCTechnicalTrigger::endJob() 
{
  
  std::cout << "RPCTechnicalTrigger::endJob>" << std::endl;
  
}

void RPCTechnicalTrigger::printinfo()
{
  
  for (int k=0; k < 3; ++k )
    m_ttu[k]->printinfo();
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);
