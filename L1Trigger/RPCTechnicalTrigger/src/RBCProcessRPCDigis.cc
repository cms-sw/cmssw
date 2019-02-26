// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessRPCDigis.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"
#include "DataFormats/Common/interface/Handle.h"
#include "GeometryConstants.h"
//-----------------------------------------------------------------------------
// Implementation file for class : RBCProcessRPCDigis
//
// 2009-04-15 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------
using namespace rpctechnicaltrigger;

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCProcessRPCDigis::RBCProcessRPCDigis(  const edm::ESHandle<RPCGeometry> & rpcGeom, 
                                         const edm::Handle<RPCDigiCollection> & digiColl ) :
  m_maxBxWindow{3},
  m_debug{false}
{
  
  m_ptr_rpcGeom  = & rpcGeom;
  m_ptr_digiColl = & digiColl;
  
  m_lbin =std::make_unique<RBCLinkBoardGLSignal>( &m_data ) ;
    
  configure();
  
}

void RBCProcessRPCDigis::configure() 
{
  for( auto wheel : s_wheelid)
    m_digiCounters.emplace(wheel, wheel);
}

//=============================================================================
// Destructor
//=============================================================================
RBCProcessRPCDigis::~RBCProcessRPCDigis() {
  
} 

//=============================================================================
int RBCProcessRPCDigis::next() {
  
  //...clean up previous data contents
  
  reset();
  
  int ndigis(0);
  
  for (auto const& detUnit : *(*m_ptr_digiColl) ) {
    
    if ( m_debug ) std::cout << "looping over digis 1 ..." << std::endl;
    
    auto digiItr = detUnit.second.first;
    int bx = (*digiItr).bx();
    
    if ( abs(bx) >= m_maxBxWindow ) {
      if ( m_debug )  std::cout << "RBCProcessRPCDigis> found a bx bigger than max allowed: "
                                << bx << std::endl;
      continue;
    }
    
    const RPCDetId & id  = detUnit.first;
    const RPCRoll * roll = dynamic_cast<const RPCRoll* >( (*m_ptr_rpcGeom)->roll(id));
    
    if((roll->isForward())) {
      if( m_debug ) std::cout << "RBCProcessRPCDigis: roll is forward" << std::endl;
      continue;
    }
    
    int wheel   = roll->id().ring();                    // -2,-1,0,+1,+2
    int sector  = roll->id().sector();                  // 1 to 12 
    int layer   = roll->id().layer();                   // 1,2
    int station = roll->id().station();                 // 1-4
    int blayer  = getBarrelLayer( layer, station );     // 1 to 6
    int rollid  = id.roll();
    
    int digipos = (station * 100) + (layer * 10) + rollid;
    
    if ( (wheel == -1 || wheel == 0 || wheel == 1) && station == 2 && layer == 1 )
      digipos = 30000 + digipos;
    if ( (wheel == -2 || wheel == 2) && station == 2 && layer == 2 )
      digipos = 30000 + digipos;
    
    if ( (wheel == -1 || wheel == 0 || wheel == 1) && station == 2 && layer == 2 )
      digipos = 20000 + digipos;
    if ( (wheel == -2 || wheel == 2) && station == 2 && layer == 1 )
      digipos = 20000 + digipos;
    
    if ( m_debug ) std::cout << "Bx: "      << bx      << '\t'
                             << "Wheel: "   << wheel   << '\t'
                             << "Sector: "  << sector  << '\t'
                             << "Station: " << station << '\t'
                             << "Layer: "   << layer   << '\t'
                             << "B-Layer: " << blayer  << '\t'
                             << "Roll id: " << rollid  << '\t'
                             << "Digi at: " << digipos << '\n';
    
    //... Construct the RBCinput objects
    auto itr = m_vecDataperBx.find( bx );
    
    if ( itr == m_vecDataperBx.end() ) {
      if ( m_debug ) std::cout << "Found a new Bx: " << bx << std::endl;
      auto& wheelData = m_vecDataperBx[bx];
      initialize(wheelData);
      setDigiAt( sector, digipos,  wheelData[ (wheel + 2) ] );
    }
    else{
      setDigiAt( sector, digipos, (*itr).second[ (wheel + 2) ] );
    }
    
    auto wheelCounter = m_digiCounters.find( wheel );
    
    if ( wheelCounter != m_digiCounters.end() )
      (*wheelCounter).second.incrementSector( sector );
    
    if ( m_debug ) std::cout << "looping over digis 2 ..." << std::endl;
    
    ++ndigis;
    
  }
  
  if ( m_debug ) std::cout << "size of data vectors: " << m_vecDataperBx.size() << std::endl;
  
  builddata();
  
  if ( m_debug ) {
    std::cout << "after reset" << std::endl;
    print_output();
  }
  
  if ( m_debug ) std::cout << "RBCProcessRPCDigis: DataSize: " << m_data.size() 
                           << " ndigis " << ndigis << std::endl;
  
  for( auto& wheelCounter : m_digiCounters) {
    wheelCounter.second.evalCounters();
    if ( m_debug ) wheelCounter.second.printSummary();
  }
  
  if ( m_data.empty() ) return 0;
  
  return 1;
  
}

void RBCProcessRPCDigis::reset()
{  
  m_vecDataperBx.clear();
}


void RBCProcessRPCDigis::initialize( std::vector<RPCData> & dataVec ) const
{
  
  if ( m_debug ) std::cout << "initialize" << std::endl;
  
  constexpr int maxWheels = 5;
  constexpr int maxRbcBrds = 6;
  
  dataVec.reserve(maxWheels);
  for(int i=0; i < maxWheels; ++i) {
    auto& block = dataVec.emplace_back();
    
    block.m_wheel = s_wheelid[i];
    
    for(int j=0; j < maxRbcBrds; ++j) {
      block.m_sec1[j] = s_sec1id[j];
      block.m_sec2[j] = s_sec2id[j];
      block.m_orsignals[j].input_sec[0].reset();
      block.m_orsignals[j].input_sec[1].reset();
      block.m_orsignals[j].needmapping = false;
      block.m_orsignals[j].hasData = false;
    }
  }
  
  if ( m_debug ) std::cout << "initialize: completed" << std::endl;
  
}

void RBCProcessRPCDigis::builddata() 
{
  
  
  for(auto & vecData: m_vecDataperBx) {
    
    int bx = vecData.first;
    int bxsign(1);
    
    if ( bx != 0 ) bxsign = ( bx / abs(bx) );
    else bxsign = 1;
    
    for(auto & item : vecData.second) {
      
      for(int k=0; k < 6; ++k) {
        
        int code = bxsign * ( 1000000*abs(bx)
                              + 10000*item.wheelIdx()
                              + 100  *item.m_sec1[k]
                              + 1    *item.m_sec2[k] );
        
        RBCInput * signal = & item.m_orsignals[k];
        signal->needmapping = false;
        
        if ( signal->hasData )
          m_data.emplace( code , signal );
        
      }
    }
  }
  
  if ( m_debug and not m_vecDataperBx.empty() ) std::cout << "builddata: completed. size of data: " << m_data.size() << std::endl;
  
}

int RBCProcessRPCDigis::getBarrelLayer( const int & _layer, const int & _station )
{
  
  //... Calculates the generic Barrel Layer (1 to 6)
  int blayer(0);
  
  if ( _station < 3 ) {
    blayer = ( (_station-1) * 2 ) + _layer;
  }
  else {
    blayer = _station + 2;
  }
  
  return blayer;
  
}


void RBCProcessRPCDigis::setDigiAt( int sector, int digipos, RPCData& block )
{
  
  int pos   = 0;
  int isAoB = 0;

  if ( m_debug ) std::cout << "setDigiAt" << std::endl;
  
  auto itr = std::find( s_sec1id.begin(), s_sec1id.end(), sector );
  
  if ( itr == s_sec1id.end()) {
    itr = std::find( s_sec2id.begin(), s_sec2id.end(), sector );
    isAoB = 1;
  } 
  
  for ( pos = 0; pos < 6; ++pos ) {
    if (block.m_sec1[pos] == sector || block.m_sec2[pos] == sector )
      break;
  }
  
  if ( m_debug ) std::cout << block.m_orsignals[pos];
  
  setInputBit( block.m_orsignals[pos].input_sec[ isAoB ] , digipos );
  
  block.m_orsignals[pos].hasData = true;
  
  if ( m_debug ) std::cout << block.m_orsignals[pos];
  
  if ( m_debug ) std::cout << "setDigiAt completed" << std::endl;
  
}

void RBCProcessRPCDigis::setInputBit( std::bitset<15> & signals , int digipos ) 
{
  
  int bitpos = s_layermap.at(digipos);
  if( m_debug ) std::cout << "Bitpos: " << bitpos << std::endl;
  signals.set( bitpos , true );
  
}

void  RBCProcessRPCDigis::print_output() 
{

  std::cout << "RBCProcessRPCDigis> Output starts" << std::endl;
  
  for( auto const& item: m_data) {
    std::cout << item.first << '\t' << (* item.second ) << '\n';
  }

  std::cout << "RBCProcessRPCDigis> Output ends" << std::endl;
  
}

