// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessRPCSimDigis.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"
#include "DataFormats/Common/interface/Handle.h"
#include "GeometryConstants.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCProcessRPCSimDigis
//
// 2009-09-20 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------
using namespace rpctechnicaltrigger;


RBCProcessRPCSimDigis::RBCProcessRPCSimDigis( const edm::ESHandle<RPCGeometry> & rpcGeom, 
                                              const edm::Handle<edm::DetSetVector<RPCDigiSimLink> > & digiSimLink)
{
  
  m_ptr_rpcGeom  = & rpcGeom;
  m_ptr_digiSimLink = & digiSimLink;
  
  m_lbin = std::make_unique<RBCLinkBoardGLSignal>( &m_data ) ;
  
  m_debug = false;
  m_maxBxWindow = 3;
  
}

//=============================================================================
// Destructor
//=============================================================================
RBCProcessRPCSimDigis::~RBCProcessRPCSimDigis() {
  reset();
} 

//=============================================================================
int RBCProcessRPCSimDigis::next() {
  
  //...clean up previous data contents
  
  reset();
  
  int ndigis(0);

  for( m_linkItr = (*m_ptr_digiSimLink)->begin();
       m_linkItr != (*m_ptr_digiSimLink)->end();
       ++m_linkItr ) {
    
    for ( m_digiItr = m_linkItr->data.begin();
          m_digiItr != m_linkItr->data.end();
          ++m_digiItr ) {
      
      if ( m_debug ) std::cout << "looping over digis 1 ..." << std::endl;
      
      int bx = (*m_digiItr).getBx();
      
      if ( abs(bx) >= m_maxBxWindow ) {
        if ( m_debug )  std::cout << "RBCProcessRPCSimDigis> found a bx bigger than max allowed: "
                                  << bx << std::endl;
        continue;
      }
      
      uint32_t detid = m_digiItr->getDetUnitId();
      const RPCDetId id( detid );
      const RPCRoll * roll = dynamic_cast<const RPCRoll* >( (*m_ptr_rpcGeom)->roll(id));
      
      if((roll->isForward())) {
        if( m_debug ) std::cout << "RBCProcessRPCSimDigis: roll is forward" << std::endl;
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
        auto& block = wheelData[ (wheel + 2) ];
        setDigiAt( sector, digipos, block );
      }
      else{
        auto& block = (*itr).second[ (wheel + 2) ];
        setDigiAt( sector, digipos, block );
      }
      
      if ( m_debug ) std::cout << "looping over digis 2 ..." << std::endl;
      
      ++ndigis;
      
    }
    
  }
  
  if ( m_debug ) std::cout << "size of data vectors: " << m_vecDataperBx.size() << std::endl;
  
  builddata();
  
  if ( m_debug ) {
    std::cout << "after reset" << std::endl;
    print_output();
  }
  
  if ( m_debug ) std::cout << "RBCProcessRPCSimDigis: DataSize: " << m_data.size() 
                           << " ndigis " << ndigis << std::endl;
  
  if ( m_data.empty() ) return 0;
  
  return 1;
  
}

void RBCProcessRPCSimDigis::reset()
{
  
  m_vecDataperBx.clear();
  
}


void RBCProcessRPCSimDigis::initialize( std::vector<RPCData> & dataVec ) 
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

void RBCProcessRPCSimDigis::builddata() 
{
  
  int code(0);
  
  for(auto& dataPerBx: m_vecDataperBx) {
    
    int bx = dataPerBx.first;
    
    int bxsign;
    if ( bx != 0 ) bxsign = ( bx / abs(bx) );
    else bxsign = 1;
    
    for(auto& item : dataPerBx.second) {
      
      for(int k=0; k < 6; ++k) {
        
        code = bxsign * ( 1000000*abs(bx)
                          + 10000*item.wheelIdx()
                          + 100  *item.m_sec1[k]
                          + 1    *item.m_sec2[k] );

        
        RBCInput * signal = & item.m_orsignals[k];
        signal->needmapping = false;
        
        if ( signal->hasData )
          m_data.insert( std::make_pair( code , signal) );
        
      }
    }
  }
  
  if ( m_debug and not m_vecDataperBx.empty()) std::cout << "builddata: completed. size of data: " << m_data.size() << std::endl;
  
}

int RBCProcessRPCSimDigis::getBarrelLayer( const int & _layer, const int & _station )
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


void RBCProcessRPCSimDigis::setDigiAt( int sector, int digipos, RPCData& block )
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

void RBCProcessRPCSimDigis::setInputBit( std::bitset<15> & signals , int digipos ) 
{
  
  int bitpos = s_layermap.at(digipos);
  if( m_debug ) std::cout << "Bitpos: " << bitpos << std::endl;
  signals.set( bitpos , true );
  
}

void  RBCProcessRPCSimDigis::print_output() 
{

  std::cout << "RBCProcessRPCSimDigis> Output starts" << std::endl;
  
  std::map<int,RBCInput*>::const_iterator itr;
  for( itr = m_data.begin(); itr != m_data.end(); ++itr) {
    std::cout << (*itr).first << '\t' << (* (*itr).second ) << '\n';
  }

  std::cout << "RBCProcessRPCSimDigis> Output ends" << std::endl;
  
}

