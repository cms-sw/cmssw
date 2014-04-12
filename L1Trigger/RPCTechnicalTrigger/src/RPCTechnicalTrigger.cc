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
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessRPCSimDigis.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================

RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig) {
  
  //...........................................................................
  
  std::string configFile  = iConfig.getParameter<std::string>("ConfigFile");
  m_verbosity             = iConfig.getUntrackedParameter<int>("Verbosity", 0);
  m_rpcDigiLabel          = iConfig.getParameter<edm::InputTag>("RPCDigiLabel");
  m_ttBits                = iConfig.getParameter< std::vector<unsigned> >("BitNumbers");
  m_ttNames               = iConfig.getParameter< std::vector<std::string> >("BitNames");
  m_useEventSetup         = iConfig.getUntrackedParameter<int>("UseEventSetup", 0);
  m_useRPCSimLink         = iConfig.getUntrackedParameter<int>("UseRPCSimLink", 0);
  m_rpcSimLinkInstance    = iConfig.getParameter<edm::InputTag>("RPCSimLinkInstance");

  edm::FileInPath f1("L1Trigger/RPCTechnicalTrigger/data/" + configFile);
  m_configFile = f1.fullPath();

  if ( m_verbosity ) {
    LogTrace("RPCTechnicalTrigger")
      << m_rpcDigiLabel << '\n'
      << std::endl;

    LogTrace("RPCTechnicalTrigger")
      << "\nConfiguration file used for UseEventSetup = 0 \n" << m_configFile << '\n'
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

  //... This is second line that delivers in parallel a second trigger
  m_ttuRbcLine[0] = new TTUEmulator( m_boardIndex[0] , m_nWheels[0] );
  m_ttuRbcLine[1] = new TTUEmulator( m_boardIndex[1] , m_nWheels[1] );
  m_ttuRbcLine[2] = new TTUEmulator( m_boardIndex[2] , m_nWheels[2] );
  
  m_WheelTtu[-2] = 3;
  m_WheelTtu[-1] = 3;
  m_WheelTtu[0 ] = 2;
  m_WheelTtu[1 ] = 1;
  m_WheelTtu[2 ] = 1;
  
  //...........................................................................
  //For the pointing Logic: declare here the first sector of each quadrant
  //
  m_quadrants.push_back(2);
  m_quadrants.push_back(3);
  m_quadrants.push_back(4);
  m_quadrants.push_back(5);
  m_quadrants.push_back(6);
  m_quadrants.push_back(7);
  m_quadrants.push_back(8);
  m_quadrants.push_back(9);
  m_quadrants.push_back(10);
  m_quadrants.push_back(11);

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
    
    delete m_ttuRbcLine[0];
    delete m_ttuRbcLine[1];
    delete m_ttuRbcLine[2];
    
    if ( m_readConfig )
      delete m_readConfig;
    
  }
  
  m_WheelTtu.clear();
    
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger: object deleted" << '\n';
  
}

//=============================================================================
void RPCTechnicalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  bool status(false);
  
  edm::Handle<RPCDigiCollection> pIn;
  
  edm::Handle<edm::DetSetVector<RPCDigiSimLink> > simIn;
  
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  
  if ( m_useRPCSimLink == 0 ) {

    iEvent.getByLabel(m_rpcDigiLabel, pIn);
    if ( ! pIn.isValid() ) {
      edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label: " 
                                           << m_rpcDigiLabel << '\n';
      iEvent.put(output);
      return;
    }
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RBCProcessRPCDigis( m_rpcGeometry, pIn ));
    
  } else {
    
    iEvent.getByLabel("simMuonRPCDigis", "RPCDigiSimLink", simIn);
    
    if ( ! simIn.isValid() ) {
      edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label: " 
                                           << m_rpcDigiLabel << '\n';
      iEvent.put(output);
      return;
    }
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RBCProcessRPCSimDigis( m_rpcGeometry, simIn ));
  }
  
  LogDebug("RPCTechnicalTrigger") << "signal object created" << '\n';
  
  if ( ! m_hasConfig ) {
    edm::LogError("RPCTechnicalTrigger") << "cannot read hardware configuration \n";
    iEvent.put(output);
    return;
  }
  
  status = m_signal->next();
  
  if ( !status)  { 
    delete m_signal;
    iEvent.put(output);
    return;
  }
  
  m_input = m_signal->retrievedata();
  
  std::vector<L1GtTechnicalTrigger> ttVec( m_ttBits.size() );
  
  //. distribute data to different TTU emulator instances and process it
  
  m_triggerbits.reset();
  
  std::vector<TTUEmulator::TriggerResponse*>::const_iterator outItr;
  
  for(int k=0; k < m_maxTtuBoards; ++k) {
    
    m_ttu[k]->processTtu( m_input );
    
    //work out Pointing Logic to Tracker
    for( m_firstSector = m_quadrants.begin(); m_firstSector != m_quadrants.end(); ++m_firstSector)
      m_ttuRbcLine[k]->processTtu( m_input , (*m_firstSector) );
    
    //...for trigger 1
    for( outItr  = m_ttu[k]->m_triggerBxVec.begin(); outItr != m_ttu[k]->m_triggerBxVec.end(); ++outItr )
      m_serializedInfoLine1.push_back( new TTUResults( k, (*outItr)->m_bx, (*outItr)->m_trigger[0], (*outItr)->m_trigger[1] ) );
    m_ttu[k]->clearTriggerResponse();
    
    //...for trigger 2
    for( outItr  = m_ttuRbcLine[k]->m_triggerBxVec.begin(); outItr != m_ttuRbcLine[k]->m_triggerBxVec.end(); ++outItr )
      m_serializedInfoLine2.push_back( new TTUResults( k, 
                                                       (*outItr)->m_bx, 
                                                       (*outItr)->m_trigger[0], 
                                                       (*outItr)->m_trigger[1], 
                                                       (*outItr)->m_wedge ) );
    
    m_ttuRbcLine[k]->clearTriggerResponse();
    
  }
  
  //.. write results to technical trigger bits
  int bx(0);
  int infoSize(0);
  
  infoSize = m_serializedInfoLine1.size();

  std::vector<RPCTechnicalTrigger::TTUResults*>::const_iterator ttuItr;
  
  std::sort( m_serializedInfoLine1.begin(), m_serializedInfoLine1.end(), sortByBx() );
  
  for( ttuItr = m_serializedInfoLine1.begin(); ttuItr != m_serializedInfoLine1.end(); ++ttuItr ) {
    if ( m_verbosity && abs( (*ttuItr)->m_bx ) <= 1 ) 
      std::cout << "RPCTechnicalTrigger> " 
                << (*ttuItr)->m_ttuidx << '\t'
                << (*ttuItr)->m_bx << '\t'
                << (*ttuItr)->m_trigWheel1 << '\t'
                << (*ttuItr)->m_trigWheel2 << '\n';
  }
  
  bool has_bx0 = false;
  
  for(int k = 0; k < infoSize; k+=m_maxTtuBoards) {
    
    bx = m_serializedInfoLine1[k]->m_bx;
    
    if ( bx == 0 ) {
      
      m_triggerbits.set(0, m_serializedInfoLine1[k]->m_trigWheel2);
      m_triggerbits.set(1, m_serializedInfoLine1[k]->m_trigWheel1);
      m_triggerbits.set(2, m_serializedInfoLine1[k+1]->m_trigWheel1);
      m_triggerbits.set(3, m_serializedInfoLine1[k+2]->m_trigWheel1);
      m_triggerbits.set(4, m_serializedInfoLine1[k+2]->m_trigWheel2);
      
      bool five_wheels_OR = m_triggerbits.any();
      
      ttVec.at(0)=L1GtTechnicalTrigger(m_ttNames.at(0), m_ttBits.at(0), bx, five_wheels_OR ) ;   // bit 24 = Or 5 wheels in TTU mode
      ttVec.at(2)=L1GtTechnicalTrigger(m_ttNames.at(2), m_ttBits.at(2), bx, m_triggerbits[0] ) ; // bit 26 
      ttVec.at(3)=L1GtTechnicalTrigger(m_ttNames.at(3), m_ttBits.at(3), bx, m_triggerbits[1] ) ; // bit 27 
      ttVec.at(4)=L1GtTechnicalTrigger(m_ttNames.at(4), m_ttBits.at(4), bx, m_triggerbits[2] ) ; // bit 28 
      ttVec.at(5)=L1GtTechnicalTrigger(m_ttNames.at(5), m_ttBits.at(5), bx, m_triggerbits[3] ) ; // bit 29
      ttVec.at(6)=L1GtTechnicalTrigger(m_ttNames.at(6), m_ttBits.at(6), bx, m_triggerbits[4] ) ; // bit 30
      
      m_triggerbits.reset();
      
      has_bx0 = true;
      
      break;
      
    } else continue;
    
  }
  
  infoSize = m_serializedInfoLine2.size();
  
  std::sort( m_serializedInfoLine2.begin(), m_serializedInfoLine2.end(), sortByBx() );
  
  for( ttuItr = m_serializedInfoLine2.begin(); ttuItr != m_serializedInfoLine2.end(); ++ttuItr ) {
    if ( m_verbosity && abs ( (*ttuItr)->m_bx ) <= 1 )
      std::cout << "RPCTechnicalTrigger> " 
                << (*ttuItr)->m_ttuidx << '\t'
                << (*ttuItr)->m_bx << '\t'
                << (*ttuItr)->m_trigWheel1 << '\t'
                << (*ttuItr)->m_trigWheel2 << '\t'
                << (*ttuItr)->m_wedge << '\n';
  }
  
  infoSize = convertToMap( m_serializedInfoLine2 );
  
  std::bitset<8> triggerCoincidence;
  triggerCoincidence.reset();
  
  // searchCoincidence( W-2 , W0 )
  bool result = searchCoincidence( -2, 0 );
  triggerCoincidence.set(0, result );
  
  // searchCoincidence( W-2 , W+1 )
  result = searchCoincidence( -2, 1 );
  triggerCoincidence.set(1, result );
  
  // searchCoincidence( W-1 , W0  )
  result = searchCoincidence( -1, 0 );
  triggerCoincidence.set(2, result );
  
  // searchCoincidence( W-1 , W+1 )
  result = searchCoincidence( -1, 1 );
  triggerCoincidence.set(3, result );
  
  // searchCoincidence( W-1 , W+2 )
  result = searchCoincidence( -1, 2 );
  triggerCoincidence.set(4, result );
  
  // searchCoincidence( W0  , W0  )
  result = searchCoincidence( 0 , 0 );
  triggerCoincidence.set(5, result );
  
  // searchCoincidence( W+1 , W0  )
  result = searchCoincidence( 1, 0 );
  triggerCoincidence.set(6, result );
  
  // searchCoincidence( W+2 , W0  ) 
  result = searchCoincidence( 2, 0 );
  triggerCoincidence.set(7, result );
  
  bool five_wheels_OR = triggerCoincidence.any();

  if ( m_verbosity ) std::cout << "RPCTechnicalTrigger> pointing trigger: " << five_wheels_OR << '\n';
  
  ttVec.at(1)=L1GtTechnicalTrigger(m_ttNames.at(1), m_ttBits.at(1), bx, five_wheels_OR ) ; // bit 25 = Or 5 wheels in RBC mode
  
  triggerCoincidence.reset();
  
  //...check that data appeared at bx=0
  
  if ( ! has_bx0 ) {
    iEvent.put(output);
    status = Reset();
    ++m_ievt;
    LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
    return;
    
  }
  
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(output);
  
  //.... all done
  
  status = Reset();
  ++m_ievt;
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
  
}

bool RPCTechnicalTrigger::Reset()
{
  
  m_input->clear();
  m_triggerbits.reset();
  std::vector<TTUResults*>::iterator itrRes;
  
  for( itrRes=m_serializedInfoLine1.begin(); itrRes!=m_serializedInfoLine1.end(); ++itrRes)
    delete (*itrRes);
  
  for( itrRes=m_serializedInfoLine2.begin(); itrRes!=m_serializedInfoLine2.end(); ++itrRes)
    delete (*itrRes);
  
  m_serializedInfoLine1.clear();
  m_serializedInfoLine2.clear();
  m_ttuResultsByQuadrant.clear();
  
  delete m_signal; 
  
  return true;
  
}

// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginRun(edm::Run const& iRun, const edm::EventSetup& evtSetup)
{
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::beginRun> starts" << std::endl;
  
  //.   Set up RPC geometry
  
  evtSetup.get<MuonGeometryRecord>().get( m_rpcGeometry );
  
  //..  Get Board Specifications (hardware configuration)
  
  if ( m_useEventSetup >= 1 ) {
    
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
    m_readConfig = new TTUConfigurator( m_configFile );
    
    if ( m_readConfig->m_hasConfig ) {
      m_readConfig->process();
      m_rbcspecs = m_readConfig->getRbcSpecs();
      m_ttuspecs = m_readConfig->getTtuSpecs();
      m_hasConfig = true;
    }
    
    else m_hasConfig = false;
    
  }
  
  if ( m_hasConfig ) {
    
    //... Initialize all
    
    for (int k=0; k < m_maxTtuBoards; ++k ) {

      m_ttu[k]->SetLineId ( 1 );
      m_ttuRbcLine[k]->SetLineId( 2 );
      
      m_ttu[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
      m_ttuRbcLine[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
      
      m_ttu[k]->initialise();
      m_ttuRbcLine[k]->initialise();
    }
  
  }
    
}

//
int RPCTechnicalTrigger::convertToMap( const std::vector<TTUResults*> & ttuResults )
{
  
  std::vector<TTUResults*>::const_iterator itr = ttuResults.begin();
  
  while ( itr != ttuResults.end() ) {
    
    if ( (*itr)->m_bx != 0 ) {
      ++itr;
      continue;
    }
    
    int key(0);
    key = 1000 * ( (*itr)->m_ttuidx + 1 ) + 1*(*itr)->m_wedge;
    m_ttuResultsByQuadrant[ key ] = (*itr);
    ++itr;
    
  }
  
  return m_ttuResultsByQuadrant.size();
    
}

//...RBC pointing logic to tracker bit 25: hardwired
bool RPCTechnicalTrigger::searchCoincidence( int wheel1, int wheel2 )
{
  
  std::map<int, TTUResults*>::iterator itr;
  bool topRight(false);
  bool botLeft(false);
  
  int indxW1 = m_WheelTtu[wheel1];
  int indxW2 = m_WheelTtu[wheel2];
  
  int k(0);
  int key(0);
  bool finalTrigger(false);
  int maxTopQuadrants = 4;
  
  //work out Pointing Logic to Tracker
  
  for( m_firstSector = m_quadrants.begin(); m_firstSector != m_quadrants.end(); ++m_firstSector) {
    
    key = 1000 * ( indxW1 ) + (*m_firstSector);
    
    itr = m_ttuResultsByQuadrant.find( key );
    if ( itr != m_ttuResultsByQuadrant.end() )
      topRight  =  (*itr).second->getTriggerForWheel(wheel1);

    //std::cout << "W1: " << wheel1 << " " << "sec: " << (*m_firstSector) << " dec: " << topRight << '\n';
    
    key = 1000 * ( indxW2 ) + (*m_firstSector) + 5;
    
    itr = m_ttuResultsByQuadrant.find( key );
    
    if ( itr != m_ttuResultsByQuadrant.end() )
      botLeft   = (*itr).second->getTriggerForWheel(wheel2);
    
    //std::cout << "W2: " << wheel2 << " " << "sec: " << (*m_firstSector) + 5 << " dec: " << botLeft << '\n';
    
    finalTrigger |= ( topRight && botLeft );
    
    ++k;
    
    if ( k > maxTopQuadrants)
      break;
        
  }
  
  //Try the opposite now

  k=0;
  
  for( m_firstSector = m_quadrants.begin(); m_firstSector != m_quadrants.end(); ++m_firstSector) {
    
    key = 1000 * ( indxW2 ) + (*m_firstSector);
    
    itr = m_ttuResultsByQuadrant.find( key );
    if ( itr != m_ttuResultsByQuadrant.end() )
      topRight  =  (*itr).second->getTriggerForWheel(wheel1);

    //std::cout << "W1: " << wheel1 << " " << "sec: " << (*m_firstSector) << " dec: " << topRight << '\n';
    
    key = 1000 * ( indxW1 ) + (*m_firstSector) + 5;
    
    itr = m_ttuResultsByQuadrant.find( key );
    
    if ( itr != m_ttuResultsByQuadrant.end() )
      botLeft   = (*itr).second->getTriggerForWheel(wheel2);
    
    //std::cout << "W2: " << wheel2 << " " << "sec: " << (*m_firstSector) + 5 << " dec: " << botLeft << '\n';
    
    finalTrigger |= ( topRight && botLeft );
    
    ++k;
    
    if ( k > maxTopQuadrants)
      break;
        
  }
  
  return finalTrigger;
  
}

// ------------ method called once each job just after ending the event loop  ------------

void RPCTechnicalTrigger::endJob() 
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::endJob>" << std::endl;
  
}

void RPCTechnicalTrigger::printinfo()
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::Printing TTU emulators info>" << std::endl;
  
  for (int k=0; k < m_maxTtuBoards; ++k ) {
    m_ttu[k]->printinfo();
    m_ttuRbcLine[k]->printinfo();
  }
  
    
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);
