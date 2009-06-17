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
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessRPCDigis.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig) {
  
  //...........................................................................
  
  m_configFile   = iConfig.getUntrackedParameter<std::string>("ConfigFile", std::string("hardware-pseudoconfig.txt"));
  
  m_verbosity    = iConfig.getUntrackedParameter<int>("Verbosity", 0);
  m_rpcDigiLabel = iConfig.getParameter<edm::InputTag>("RPCDigiLabel");
  m_ttBits       = iConfig.getParameter< std::vector<unsigned> >("BitNumbers");
  m_ttNames      = iConfig.getParameter< std::vector<std::string> >("BitNames");
  m_useDatabase  = iConfig.getUntrackedParameter<int>("UseDatabase", 1);
  
  if ( m_verbosity ) {
    LogTrace("RPCTechnicalTrigger")
      << m_rpcDigiLabel << '\n'
      << std::endl;
  }
  
  //...........................................................................
  //... There are three Technical Trigger Units Boards: 1 can handle 2 Wheels
  //... n_Wheels sets the number of wheels attached to board with index boardIndex
  
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
  m_hasConfig = false;
  m_readConfig = NULL;
  produces<L1GtTechnicalTriggerRecord>();
  
}


RPCTechnicalTrigger::~RPCTechnicalTrigger()
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger: object starts deletion" << std::endl;

  if ( m_hasConfig ) {
    
    delete m_ttu[0];
    delete m_ttu[1];
    delete m_ttu[2];
    
    if ( m_readConfig )
      delete m_readConfig;
    
  }
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger: object deleted" << '\n';
  
}

//=============================================================================
void RPCTechnicalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  bool status(false);
  
  edm::Handle<RPCDigiCollection> pIn;
  
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  
  iEvent.getByLabel(m_rpcDigiLabel, pIn);

  if ( ! m_hasConfig ) {
    edm::LogError("RPCTechnicalTrigger") << "cannot read hardware configuration \n";
    iEvent.put(output);
    return;
  }
  
  if ( !pIn.isValid() ) {
    edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label: " 
                                         << m_rpcDigiLabel << '\n';
    iEvent.put(output);
    return;
  }
  
  
  LogDebug("RPCTechnicalTrigger") << "Trigger mode 0: Ttu only" << '\n';

  m_signal  = dynamic_cast<ProcessInputSignal*>(new RBCProcessRPCDigis( m_rpcGeometry, pIn ));
  
  LogDebug("RPCTechnicalTrigger") << "signal object created" << '\n';
  
  status = m_signal->next();
  
  if ( !status)  { 
    delete m_signal;
    iEvent.put(output);
    return;
  }
  
  RPCInputSignal * input = m_signal->retrievedata();
  
  std::vector<L1GtTechnicalTrigger> ttVec( m_ttBits.size() );
  
  //. distribute data to different TTU emulators and process it
  
  m_triggerbits.reset();
  
  int indx(0);
  
  std::vector<TTUEmulator::TriggerResponse*>::const_iterator outItr;
  
  for(int k=0; k < m_maxTtuBoards; ++k) {
    
    indx=k*2;
    
    m_ttu[k]->processTtu( input );
    
    for( outItr  = m_ttu[k]->m_triggerBxVec.begin(); outItr != m_ttu[k]->m_triggerBxVec.end(); ++outItr )
      m_serializedInfo.push_back( new TTUResults( k, (*outItr)->m_bx, (*outItr)->m_trigger[0], (*outItr)->m_trigger[1] ) );
    m_ttu[k]->clearTriggerResponse();
    
  }
  
  //.. write results to technical trigger bits
  int bx(0);
  int infoSize(0);
  
  infoSize = m_serializedInfo.size();
  std::sort( m_serializedInfo.begin(), m_serializedInfo.end(), sortByBx() );
  
  for(int k = 0; k < infoSize; k+=m_maxTtuBoards) {
    
    m_triggerbits.set(0, m_serializedInfo[k]->m_trigWheel1);
    m_triggerbits.set(1, m_serializedInfo[k]->m_trigWheel2);
    m_triggerbits.set(2, m_serializedInfo[k+1]->m_trigWheel1);
    m_triggerbits.set(3, m_serializedInfo[k+2]->m_trigWheel1);
    m_triggerbits.set(4, m_serializedInfo[k+2]->m_trigWheel2);
    
    bx = m_serializedInfo[k]->m_bx;
    
    for(int i = 0; i < m_maxBits; ++i) {
      ttVec.at(i)=L1GtTechnicalTrigger(m_ttNames.at(i), m_ttBits.at(i), bx, m_triggerbits[i] ) ;
    }
    
    m_triggerbits.reset();
    
  }
  
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(output);
  
  //... reset data map for next event

  input->clear();

  m_triggerbits.reset();
  
  std::vector<TTUResults*>::iterator itrRes;

  for( itrRes=m_serializedInfo.begin(); itrRes!=m_serializedInfo.end(); ++itrRes)
    delete (*itrRes);

  m_serializedInfo.clear();

  delete m_signal;
  
  //.... all done
  
  ++m_ievt;

  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
  
}

// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginRun(edm::Run& iRun, const edm::EventSetup& evtSetup)
{
  
  bool status(false);
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::beginRun> starts" << std::endl;
  
  //.   Set up RPC geometry
  
  evtSetup.get<MuonGeometryRecord>().get( m_rpcGeometry );
  
  //..  Get Board Specifications (hardware configuration)
  
  if ( m_useDatabase >= 1 ) {
    
    edm::ESHandle<RBCBoardSpecs> pRBCSpecs;
    evtSetup.get<RBCBoardSpecsRcd>().get(pRBCSpecs);

    edm::ESHandle<TTUBoardSpecs> pTTUSpecs;
    evtSetup.get<TTUBoardSpecsRcd>().get(pTTUSpecs);
    
    if ( !pRBCSpecs.isValid() ||  !pTTUSpecs.isValid() ) {
      edm::LogError("RPCTechnicalTrigger") << "can't find RBC/TTU BoardSpecsRcd" << '\n';
      m_hasConfig = false;
    }
    else  {
      m_rbcspecs = pRBCSpecs.product();
      m_ttuspecs = pTTUSpecs.product();
      m_hasConfig = true;
    }
    
  } else {
    
    // read hardware configuration from file
    m_readConfig = new TTUConfigurator( m_configFile.c_str() );
    
    if ( m_readConfig->m_hasConfig ) {
      m_readConfig->process();
      m_rbcspecs = m_readConfig->getRbcSpecs();
      m_ttuspecs = m_readConfig->getTtuSpecs();
      m_hasConfig = true;
    }
    
    else m_hasConfig = false;
    
  }
  
  if ( m_hasConfig ) {
    

    for (int k=0; k < m_maxTtuBoards; ++k )
      m_ttu[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
    
    //... Initialize all
    
    for (int k=0; k < m_maxTtuBoards; ++k )
      status = m_ttu[k]->initialise();
    
  }
  
  
}

// ------------ method called once each job just after ending the event loop  ------------

void RPCTechnicalTrigger::endJob() 
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::endJob>" << std::endl;
  
}

void RPCTechnicalTrigger::printinfo()
{

  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::Printing TTU emulators info>" << std::endl;
  
  for (int k=0; k < m_maxTtuBoards; ++k )
    m_ttu[k]->printinfo();
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);
