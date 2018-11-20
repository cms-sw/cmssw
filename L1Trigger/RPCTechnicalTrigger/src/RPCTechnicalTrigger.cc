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

namespace {
  //...........................................................................
  //For the pointing Logic: declare here the first sector of each quadrant
  //
  constexpr std::array<int,10> s_quadrants = { {2,3,4,5,6,7,8,9,10,11} };

  //The wheelTtu is addressed using -2, -1, 0, 1, 2
  constexpr unsigned int kWheelOffset = 2;
  constexpr std::array<int, 5> wheelTtu = { {3,3,2,1,1} };
}

RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig):
  m_verbosity{ iConfig.getUntrackedParameter<int>("Verbosity", 0)},
  m_useEventSetup{iConfig.getUntrackedParameter<int>("UseEventSetup", 0)},
  m_ttBits{iConfig.getParameter< std::vector<unsigned> >("BitNumbers")},
  m_ttNames{iConfig.getParameter< std::vector<std::string> >("BitNames")},
  m_rpcDigiLabel{iConfig.getParameter<edm::InputTag>("RPCDigiLabel")},
  m_rpcDigiToken{consumes<RPCDigiCollection>(m_rpcDigiLabel)},
  m_useRPCSimLink{iConfig.getUntrackedParameter<int>("UseRPCSimLink", 0)}
{
  
  //...........................................................................
  
  std::string configFile  = iConfig.getParameter<std::string>("ConfigFile");

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
  
  constexpr std::array<int,3> boardIndex={{1,2,3}};
  constexpr std::array<int,3> nWheels = { {2,1,2} };

  m_ttu[0] = TTUEmulator( boardIndex[0] , nWheels[0] );
  m_ttu[1] = TTUEmulator( boardIndex[1] , nWheels[1] );
  m_ttu[2] = TTUEmulator( boardIndex[2] , nWheels[2] );

  //... This is second line that delivers in parallel a second trigger
  m_ttuRbcLine[0] = TTUEmulator( boardIndex[0] , nWheels[0] );
  m_ttuRbcLine[1] = TTUEmulator( boardIndex[1] , nWheels[1] );
  m_ttuRbcLine[2] = TTUEmulator( boardIndex[2] , nWheels[2] );
  
  //...........................................................................
  
  m_hasConfig = false;
  produces<L1GtTechnicalTriggerRecord>();
  consumes<edm::DetSetVector<RPCDigiSimLink> >(edm::InputTag("simMuonRPCDigis", "RPCDigiSimLink",""));
}


RPCTechnicalTrigger::~RPCTechnicalTrigger()
{  
}

//=============================================================================
void RPCTechnicalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  bool status(false);
  
  edm::Handle<RPCDigiCollection> pIn;
  
  edm::Handle<edm::DetSetVector<RPCDigiSimLink> > simIn;
  
  std::unique_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  
  //.   Set up RPC geometry
  edm::ESHandle<RPCGeometry> rpcGeometry;
  iSetup.get<MuonGeometryRecord>().get( rpcGeometry );

  std::unique_ptr<ProcessInputSignal> signal;
  if ( m_useRPCSimLink == 0 ) {
    iEvent.getByToken(m_rpcDigiToken, pIn);
    if ( ! pIn.isValid() ) {
      edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label: " 
                                           << m_rpcDigiLabel << '\n';
      iEvent.put(std::move(output));
      return;
    }
  
    signal  = std::make_unique<RBCProcessRPCDigis>( rpcGeometry, pIn );
    
  } else {
    
    iEvent.getByLabel("simMuonRPCDigis", "RPCDigiSimLink", simIn);
    
    if ( ! simIn.isValid() ) {
      edm::LogError("RPCTechnicalTrigger") << "can't find RPCDigiCollection with label: " 
                                           << m_rpcDigiLabel << '\n';
      iEvent.put(std::move(output));
      return;
    }
    signal  = std::make_unique<RBCProcessRPCSimDigis>( rpcGeometry, simIn );
  }
  
  LogDebug("RPCTechnicalTrigger") << "signal object created" << '\n';
  
  if ( ! m_hasConfig ) {
    edm::LogError("RPCTechnicalTrigger") << "cannot read hardware configuration \n";
    iEvent.put(std::move(output));
    return;
  }
  
  status = signal->next();
  
  if ( !status)  { 
    iEvent.put(std::move(output));
    return;
  }
  
  auto* input = signal->retrievedata();
  
  std::vector<L1GtTechnicalTrigger> ttVec( m_ttBits.size() );
  
  //. distribute data to different TTU emulator instances and process it
  std::bitset<5> triggerbits;
  
  std::vector<std::unique_ptr<TTUResults>> serializedInfoLine1;
  std::vector<std::unique_ptr<TTUResults>> serializedInfoLine2;

  for(int k=0; k < kMaxTtuBoards; ++k) {
    
    m_ttu[k].processTtu( input );
    
    //work out Pointing Logic to Tracker
    for( auto  firstSector : s_quadrants)
      m_ttuRbcLine[k].processTtu( input , firstSector );
    
    //...for trigger 1
    for( auto const& out : m_ttu[k].m_triggerBxVec)
      serializedInfoLine1.emplace_back( std::make_unique<TTUResults>( k, out.m_bx, out.m_trigger[0], out.m_trigger[1] ) );
    m_ttu[k].clearTriggerResponse();
    
    //...for trigger 2
    for( auto const& out : m_ttuRbcLine[k].m_triggerBxVec)
      serializedInfoLine2.push_back( std::make_unique<TTUResults>( k, 
                                                                   out.m_bx, 
                                                                   out.m_trigger[0], 
                                                                   out.m_trigger[1], 
                                                                   out.m_wedge ) );
    
    m_ttuRbcLine[k].clearTriggerResponse();
    
  }
  
  //.. write results to technical trigger bits
  int bx(0);
  int infoSize(0);
  
  infoSize = serializedInfoLine1.size();

  auto sortByBx = [](auto& iLHS, auto& iRHS) {
      return iLHS->m_bx < iRHS->m_bx;
  };
  std::sort( serializedInfoLine1.begin(), serializedInfoLine1.end(), sortByBx );
  
  if( m_verbosity ) {
    for( auto& ttu : serializedInfoLine1) {
      if ( abs( ttu->m_bx ) <= 1 ) 
        std::cout << "RPCTechnicalTrigger> " 
                  << ttu->m_ttuidx << '\t'
                  << ttu->m_bx << '\t'
                  << ttu->m_trigWheel1 << '\t'
                  << ttu->m_trigWheel2 << '\n';
    }
  }
  
  bool has_bx0 = false;
  
  for(int k = 0; k < infoSize; k+=kMaxTtuBoards) {
    
    bx = serializedInfoLine1[k]->m_bx;
    
    if ( bx == 0 ) {
      
      triggerbits.set(0, serializedInfoLine1[k]->m_trigWheel2);
      triggerbits.set(1, serializedInfoLine1[k]->m_trigWheel1);
      triggerbits.set(2, serializedInfoLine1[k+1]->m_trigWheel1);
      triggerbits.set(3, serializedInfoLine1[k+2]->m_trigWheel1);
      triggerbits.set(4, serializedInfoLine1[k+2]->m_trigWheel2);
      
      bool five_wheels_OR = triggerbits.any();
      
      ttVec.at(0)=L1GtTechnicalTrigger(m_ttNames.at(0), m_ttBits.at(0), bx, five_wheels_OR ) ;   // bit 24 = Or 5 wheels in TTU mode
      ttVec.at(2)=L1GtTechnicalTrigger(m_ttNames.at(2), m_ttBits.at(2), bx, triggerbits[0] ) ; // bit 26 
      ttVec.at(3)=L1GtTechnicalTrigger(m_ttNames.at(3), m_ttBits.at(3), bx, triggerbits[1] ) ; // bit 27 
      ttVec.at(4)=L1GtTechnicalTrigger(m_ttNames.at(4), m_ttBits.at(4), bx, triggerbits[2] ) ; // bit 28 
      ttVec.at(5)=L1GtTechnicalTrigger(m_ttNames.at(5), m_ttBits.at(5), bx, triggerbits[3] ) ; // bit 29
      ttVec.at(6)=L1GtTechnicalTrigger(m_ttNames.at(6), m_ttBits.at(6), bx, triggerbits[4] ) ; // bit 30
      
      triggerbits.reset();
      
      has_bx0 = true;
      
      break;
      
    } else continue;
    
  }
  
  infoSize = serializedInfoLine2.size();
  
  std::sort( serializedInfoLine2.begin(), serializedInfoLine2.end(), sortByBx );
  
  if(m_verbosity) {
    for( auto& ttu : serializedInfoLine2) {
      if (abs ( ttu->m_bx ) <= 1 )
        std::cout << "RPCTechnicalTrigger> " 
                  << ttu->m_ttuidx << '\t'
                  << ttu->m_bx << '\t'
                  << ttu->m_trigWheel1 << '\t'
                  << ttu->m_trigWheel2 << '\t'
                  << ttu->m_wedge << '\n';
    }
  }
  
  auto ttuResultsByQuadrant  = convertToMap( serializedInfoLine2 );
  
  std::bitset<8> triggerCoincidence;
  triggerCoincidence.reset();
  
  // searchCoincidence( W-2 , W0 )
  bool result = searchCoincidence( -2, 0, ttuResultsByQuadrant );
  triggerCoincidence.set(0, result );
  
  // searchCoincidence( W-2 , W+1 )
  result = searchCoincidence( -2, 1, ttuResultsByQuadrant  );
  triggerCoincidence.set(1, result );
  
  // searchCoincidence( W-1 , W0  )
  result = searchCoincidence( -1, 0, ttuResultsByQuadrant  );
  triggerCoincidence.set(2, result );
  
  // searchCoincidence( W-1 , W+1 )
  result = searchCoincidence( -1, 1, ttuResultsByQuadrant  );
  triggerCoincidence.set(3, result );
  
  // searchCoincidence( W-1 , W+2 )
  result = searchCoincidence( -1, 2, ttuResultsByQuadrant  );
  triggerCoincidence.set(4, result );
  
  // searchCoincidence( W0  , W0  )
  result = searchCoincidence( 0 , 0, ttuResultsByQuadrant  );
  triggerCoincidence.set(5, result );
  
  // searchCoincidence( W+1 , W0  )
  result = searchCoincidence( 1, 0, ttuResultsByQuadrant  );
  triggerCoincidence.set(6, result );
  
  // searchCoincidence( W+2 , W0  ) 
  result = searchCoincidence( 2, 0, ttuResultsByQuadrant  );
  triggerCoincidence.set(7, result );
  
  bool five_wheels_OR = triggerCoincidence.any();

  if ( m_verbosity ) std::cout << "RPCTechnicalTrigger> pointing trigger: " << five_wheels_OR << '\n';
  
  ttVec.at(1)=L1GtTechnicalTrigger(m_ttNames.at(1), m_ttBits.at(1), bx, five_wheels_OR ) ; // bit 25 = Or 5 wheels in RBC mode
  
  triggerCoincidence.reset();
  
  //...check that data appeared at bx=0
  
  if ( ! has_bx0 ) {
    iEvent.put(std::move(output));
    LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
    return;
    
  }
  
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(std::move(output));
  
  //.... all done
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger> end of event loop" << std::endl;
  
}
// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginRun(edm::Run const& iRun, const edm::EventSetup& evtSetup)
{
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::beginRun> starts" << std::endl;
  
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
    m_readConfig = std::make_unique<TTUConfigurator>( m_configFile );
    
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
    
    for (int k=0; k < kMaxTtuBoards; ++k ) {

      m_ttu[k].SetLineId ( 1 );
      m_ttuRbcLine[k].SetLineId( 2 );
      
      m_ttu[k].setSpecifications( m_ttuspecs, m_rbcspecs );
      m_ttuRbcLine[k].setSpecifications( m_ttuspecs, m_rbcspecs );
      
      m_ttu[k].initialise();
      m_ttuRbcLine[k].initialise();
    }
  
  }
    
}

//
std::map<int,RPCTechnicalTrigger::TTUResults*>
RPCTechnicalTrigger::convertToMap( const std::vector<std::unique_ptr<TTUResults>> & ttuResults ) const
{
  std::map<int,TTUResults*> returnValue;
  auto itr = ttuResults.begin();
  
  while ( itr != ttuResults.end() ) {
    
    if ( (*itr)->m_bx != 0 ) {
      ++itr;
      continue;
    }
    
    int key(0);
    key = 1000 * ( (*itr)->m_ttuidx + 1 ) + 1*(*itr)->m_wedge;
    returnValue[ key ] = itr->get();
    ++itr;
  }
  
  return returnValue;
    
}

//...RBC pointing logic to tracker bit 25: hardwired
bool RPCTechnicalTrigger::searchCoincidence( int wheel1, int wheel2,  std::map<int, TTUResults*> const& ttuResultsByQuadrant) const
{
  
  std::map<int, TTUResults*>::const_iterator itr;
  bool topRight(false);
  bool botLeft(false);
  
  int indxW1 = wheelTtu[wheel1+kWheelOffset];
  int indxW2 = wheelTtu[wheel2+kWheelOffset];
  
  int k(0);
  int key(0);
  bool finalTrigger(false);
  int maxTopQuadrants = 4;
  
  //work out Pointing Logic to Tracker
  
  for( auto firstSector : s_quadrants) {
    
    key = 1000 * ( indxW1 ) + firstSector;
    
    itr = ttuResultsByQuadrant.find( key );
    if ( itr != ttuResultsByQuadrant.end() )
      topRight  =  (*itr).second->getTriggerForWheel(wheel1);

    //std::cout << "W1: " << wheel1 << " " << "sec: " << firstSector << " dec: " << topRight << '\n';
    
    key = 1000 * ( indxW2 ) + firstSector + 5;
    
    itr = ttuResultsByQuadrant.find( key );
    
    if ( itr != ttuResultsByQuadrant.end() )
      botLeft   = (*itr).second->getTriggerForWheel(wheel2);
    
    //std::cout << "W2: " << wheel2 << " " << "sec: " << firstSector + 5 << " dec: " << botLeft << '\n';
    
    finalTrigger |= ( topRight && botLeft );
    
    ++k;
    
    if ( k > maxTopQuadrants)
      break;
        
  }
  
  //Try the opposite now

  k=0;
  
  for( auto firstSector : s_quadrants) {
    
    key = 1000 * ( indxW2 ) + firstSector;
    
    itr = ttuResultsByQuadrant.find( key );
    if ( itr != ttuResultsByQuadrant.end() )
      topRight  =  (*itr).second->getTriggerForWheel(wheel1);

    //std::cout << "W1: " << wheel1 << " " << "sec: " << firstSector << " dec: " << topRight << '\n';
    
    key = 1000 * ( indxW1 ) + firstSector + 5;
    
    itr = ttuResultsByQuadrant.find( key );
    
    if ( itr != ttuResultsByQuadrant.end() )
      botLeft   = (*itr).second->getTriggerForWheel(wheel2);
    
    //std::cout << "W2: " << wheel2 << " " << "sec: " << firstSector + 5 << " dec: " << botLeft << '\n';
    
    finalTrigger |= ( topRight && botLeft );
    
    ++k;
    
    if ( k > maxTopQuadrants)
      break;
        
  }
  
  return finalTrigger;
  
}

// ------------ method called once each job just after ending the event loop  ------------

void RPCTechnicalTrigger::printinfo() const
{
  
  LogDebug("RPCTechnicalTrigger") << "RPCTechnicalTrigger::Printing TTU emulators info>" << std::endl;
  
  for (int k=0; k < kMaxTtuBoards; ++k ) {
    m_ttu[k].printinfo();
    m_ttuRbcLine[k].printinfo();
  }
  
    
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);
