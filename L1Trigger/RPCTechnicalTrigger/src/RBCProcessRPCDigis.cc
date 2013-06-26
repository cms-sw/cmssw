// $Id: RBCProcessRPCDigis.cc,v 1.6 2013/03/20 15:45:25 wdd Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessRPCDigis.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"
#include "DataFormats/Common/interface/Handle.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCProcessRPCDigis
//
// 2009-04-15 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCProcessRPCDigis::RBCProcessRPCDigis(  const edm::ESHandle<RPCGeometry> & rpcGeom, 
                                         const edm::Handle<RPCDigiCollection> & digiColl ) 
{
  
  m_ptr_rpcGeom  = & rpcGeom;
  m_ptr_digiColl = & digiColl;
  
  m_lbin = dynamic_cast<RPCInputSignal*>( new RBCLinkBoardGLSignal( &m_data ) );
  
  m_debug = false;
  
  configure();
  
}

void RBCProcessRPCDigis::configure() 
{
  
  m_wheelid.push_back(-2); //-2
  m_wheelid.push_back(-1); //-1
  m_wheelid.push_back(0);  // 0
  m_wheelid.push_back( 1); //+1
  m_wheelid.push_back( 2); //+2
  
  m_sec1id.push_back(12);
  m_sec2id.push_back(1);
  m_sec1id.push_back(2);
  m_sec2id.push_back(3);
  m_sec1id.push_back(4);
  m_sec2id.push_back(5);
  m_sec1id.push_back(6);
  m_sec2id.push_back(7);
  m_sec1id.push_back(8);
  m_sec2id.push_back(9);
  m_sec1id.push_back(10);
  m_sec2id.push_back(11);
  
  m_layermap[113]     = 0;  //RB1InFw
  m_layermap[123]     = 1;  //RB1OutFw
  
  m_layermap[20213]   = 2;  //RB22Fw
  m_layermap[20223]   = 2;  //RB22Fw
  m_layermap[30223]   = 3;  //RB23Fw
  m_layermap[30213]   = 3;  //RB23Fw
  m_layermap[30212]   = 4;  //RB23M
  m_layermap[30222]   = 4;  //RB23M
  
  m_layermap[313]     = 5;  //RB3Fw
  m_layermap[413]     = 6;  //RB4Fw
  m_layermap[111]     = 7;  //RB1InBk
  m_layermap[121]     = 8;  //RB1OutBk
  
  m_layermap[20211]   = 9;  //RB22Bw
  m_layermap[20221]   = 9;  //RB22Bw
  m_layermap[30211]   = 10; //RB23Bw
  m_layermap[30221]   = 10; //RB23Bw
  
  m_layermap[311]     = 11; //RB3Bk
  m_layermap[411]     = 12; //RB4Bk

  m_maxBxWindow = 3;

  std::vector<int>::iterator wheel;

  for( wheel = m_wheelid.begin(); wheel != m_wheelid.end(); ++wheel)
    m_digiCounters[(*wheel)] = new l1trigger::Counters( (*wheel) );
  
}

//=============================================================================
// Destructor
//=============================================================================
RBCProcessRPCDigis::~RBCProcessRPCDigis() {
  
  if ( m_lbin ) delete m_lbin;

  std::vector<int>::iterator wheel;
  
  for( wheel = m_wheelid.begin(); wheel != m_wheelid.end(); ++wheel)
    delete m_digiCounters[(*wheel)];
  
  m_sec1id.clear();
  m_sec2id.clear();
  m_wheelid.clear();
  m_layermap.clear();
  
  reset();
    
} 

//=============================================================================
int RBCProcessRPCDigis::next() {
  
  //...clean up previous data contents
  
  reset();
  
  int ndigis(0);
  
  for (m_detUnitItr = (*m_ptr_digiColl)->begin(); 
       m_detUnitItr != (*m_ptr_digiColl)->end(); ++m_detUnitItr ) {
    
    if ( m_debug ) std::cout << "looping over digis 1 ..." << std::endl;
    
    m_digiItr = (*m_detUnitItr ).second.first;
    int bx = (*m_digiItr).bx();
    
    if ( abs(bx) >= m_maxBxWindow ) {
      if ( m_debug )  std::cout << "RBCProcessRPCDigis> found a bx bigger than max allowed: "
                                << bx << std::endl;
      continue;
    }
    
    const RPCDetId & id  = (*m_detUnitItr).first;
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
    std::map<int,std::vector<RPCData*> >::iterator itr;
    itr = m_vecDataperBx.find( bx );
    
    if ( itr == m_vecDataperBx.end() ) {
      if ( m_debug ) std::cout << "Found a new Bx: " << bx << std::endl;
      std::vector<RPCData*> wheelData;
      initialize(wheelData);
      m_vecDataperBx[bx] = wheelData; 
      this->m_block = wheelData[ (wheel + 2) ];
      setDigiAt( sector, digipos );
    }
    else{
      this->m_block = (*itr).second[ (wheel + 2) ];
      setDigiAt( sector, digipos );
    }
    
    std::map<int, l1trigger::Counters* >::iterator wheelCounter;
    wheelCounter = m_digiCounters.find( wheel );
    
    if ( wheelCounter != m_digiCounters.end() )
      (*wheelCounter).second->incrementSector( sector );
    
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
  
  std::map<int, l1trigger::Counters* >::iterator wheelCounter;
  for( wheelCounter = m_digiCounters.begin(); wheelCounter != m_digiCounters.end(); ++wheelCounter) {
    (*wheelCounter).second->evalCounters();
    if ( m_debug ) (*wheelCounter).second->printSummary();
  }
  
  if ( m_data.size() <= 0 ) return 0;
  
  return 1;
  
}

void RBCProcessRPCDigis::reset()
{
  
  std::map<int,std::vector<RPCData*> >::iterator itr1;
  for( itr1 = m_vecDataperBx.begin(); itr1 != m_vecDataperBx.end(); ++itr1) {
    std::vector<RPCData*>::iterator itr2;
    for(itr2 = (*itr1).second.begin(); itr2 != (*itr1).second.end();++itr2 )
      if ( (*itr2) ) delete *itr2;
    (*itr1).second.clear();
  }
  m_vecDataperBx.clear();
  
}


void RBCProcessRPCDigis::initialize( std::vector<RPCData*> & dataVec ) 
{
  
  if ( m_debug ) std::cout << "initialize" << std::endl;
  
  int maxWheels = 5;
  int maxRbcBrds = 6;
  
  for(int i=0; i < maxWheels; ++i) {
    
    m_block = new RPCData();
    
    m_block->m_wheel = m_wheelid[i];
    
    for(int j=0; j < maxRbcBrds; ++j) {
      m_block->m_sec1[j] = m_sec1id[j];
      m_block->m_sec2[j] = m_sec2id[j];
      m_block->m_orsignals[j].input_sec[0].reset();
      m_block->m_orsignals[j].input_sec[1].reset();
      m_block->m_orsignals[j].needmapping = false;
      m_block->m_orsignals[j].hasData = false;
    }

    dataVec.push_back( m_block );
    
  }
  
  if ( m_debug ) std::cout << "initialize: completed" << std::endl;
  
}

void RBCProcessRPCDigis::builddata() 
{
  
  int bx(0);
  int code(0);
  int bxsign(1);
  std::vector<RPCData*>::iterator itr;
  std::map<int, std::vector<RPCData*> >::iterator itr2;
  
  itr2 = m_vecDataperBx.begin();
  if( itr2 == ( m_vecDataperBx.end() ) ) return;
  
  while ( itr2 != m_vecDataperBx.end() ) {
    
    bx = (*itr2).first;
    
    if ( bx != 0 ) bxsign = ( bx / abs(bx) );
    else bxsign = 1;
    
    for(itr = (*itr2).second.begin(); itr != (*itr2).second.end(); ++itr) {
      
      for(int k=0; k < 6; ++k) {
        
        code = bxsign * ( 1000000*abs(bx)
                          + 10000*(*itr)->wheelIdx()
                          + 100  *(*itr)->m_sec1[k]
                          + 1    *(*itr)->m_sec2[k] );
        
        RBCInput * signal = & (*itr)->m_orsignals[k];
        signal->needmapping = false;
        
        if ( signal->hasData )
          m_data.insert( std::make_pair( code , signal) );
        
      }
    }
    
    ++itr2;
    
  }
  
  if ( m_debug ) std::cout << "builddata: completed. size of data: " << m_data.size() << std::endl;
  
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


void RBCProcessRPCDigis::setDigiAt( int sector, int digipos )
{
  
  int pos   = 0;
  int isAoB = 0;

  if ( m_debug ) std::cout << "setDigiAt" << std::endl;
  
  std::vector<int>::const_iterator itr;
  itr = std::find( m_sec1id.begin(), m_sec1id.end(), sector );
  
  if ( itr == m_sec1id.end()) {
    itr = std::find( m_sec2id.begin(), m_sec2id.end(), sector );
    isAoB = 1;
  } 
  
  for ( pos = 0; pos < 6; ++pos ) {
    if (this->m_block->m_sec1[pos] == sector || this->m_block->m_sec2[pos] == sector )
      break;
  }
  
  if ( m_debug ) std::cout << this->m_block->m_orsignals[pos];
  
  setInputBit( this->m_block->m_orsignals[pos].input_sec[ isAoB ] , digipos );
  
  this->m_block->m_orsignals[pos].hasData = true;
  
  if ( m_debug ) std::cout << this->m_block->m_orsignals[pos];
  
  if ( m_debug ) std::cout << "setDigiAt completed" << std::endl;
  
}

void RBCProcessRPCDigis::setInputBit( std::bitset<15> & signals , int digipos ) 
{
  
  int bitpos = m_layermap[digipos];
  if( m_debug ) std::cout << "Bitpos: " << bitpos << std::endl;
  signals.set( bitpos , 1 );
  
}

void  RBCProcessRPCDigis::print_output() 
{

  std::cout << "RBCProcessRPCDigis> Output starts" << std::endl;
  
  std::map<int,RBCInput*>::const_iterator itr;
  for( itr = m_data.begin(); itr != m_data.end(); ++itr) {
    std::cout << (*itr).first << '\t' << (* (*itr).second ) << '\n';
  }

  std::cout << "RBCProcessRPCDigis> Output ends" << std::endl;
  
}

