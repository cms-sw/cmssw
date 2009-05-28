// $Id: ProcessDigiGlobalSignal.cc,v 1.5 2009/05/26 17:40:38 aosorio Exp $
// Include files 


// local
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessDigiGlobalSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUGlobalSignal.h" 

//-----------------------------------------------------------------------------
// Implementation file for class : ProcessDigiGlobalSignal 
// (RPCTechnicalTrigger Emulator
// 2008-11-23 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
ProcessDigiGlobalSignal::ProcessDigiGlobalSignal( const edm::ESHandle<RPCGeometry> & rpcGeom, 
                                                  const edm::Handle<RPCDigiCollection> & digiColl ) 
{
  
  m_ptr_rpcGeom  = & rpcGeom;
  m_ptr_digiColl = & digiColl;
  
  m_wmin = dynamic_cast<RPCInputSignal*>( new TTUGlobalSignal( &m_data ) );

  m_maxBx = 7;
  m_maxBxWindow = 3;
    
  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
ProcessDigiGlobalSignal::~ProcessDigiGlobalSignal() {
  
  if( m_wmin ) delete m_wmin;
  
  std::map<int, RPCWheelMap*>::iterator itr;
  for(itr = m_wheelMapVec.begin(); itr != m_wheelMapVec.begin(); ++itr)
    if( (*itr).second ) delete (*itr).second;
  
  m_wheelMapVec.clear();
  
} 

//=============================================================================
int ProcessDigiGlobalSignal::next() 
{
  
  //. Generate Wheel Maps from Digi collection

  reset();
    
  RPCWheelMap * wheelmap;
  
  for (m_detUnitItr = (*m_ptr_digiColl)->begin(); 
       m_detUnitItr != (*m_ptr_digiColl)->end(); ++m_detUnitItr ) {
    
    if ( m_debug ) std::cout << "looping over digis 1 ..." << std::endl;
    
    m_digiItr = (*m_detUnitItr ).second.first;
    int bx = (*m_digiItr).bx();

    if ( abs(bx) > m_maxBxWindow ) {
      if ( m_debug )  std::cout << "ProcessDigiGlobalSignal> found a bx bigger than max allowed: "
                                << bx << std::endl;
      continue;
    }
    
    const RPCDetId & id  = (*m_detUnitItr).first;
    const RPCRoll * roll = dynamic_cast<const RPCRoll* >( (*m_ptr_rpcGeom)->roll(id));
    
    if((roll->isForward())) {
      if( m_debug ) std::cout << "ProcessDigiGlobalSignal: roll is forward" << std::endl;
      continue;
    }
    
    int wheel   = roll->id().ring();                    // -2,-1,0,+1,+2
    int sector  = roll->id().sector() - 1;              // 0 to 11 (12)
    int layer   = roll->id().layer();                   // 1,2
    int station = roll->id().station();                 // 1-4
    int blayer  = getBarrelLayer( layer, station );     // 1 to 6
    
    if ( m_debug ) std::cout << "Bx: "      << bx      << '\t'
                             << "Wheel: "   << wheel   << '\t'
                             << "Sector: "  << sector  << '\t'
                             << "Station: " << station << '\t'
                             << "Layer: "   << layer   << '\t'
                             << "B-Layer: " << blayer  << '\n';
    
    
    std::map<int, RPCWheelMap*>::iterator itr;
    itr = m_wheelMapVec.find( wheel );
    
    if ( itr == m_wheelMapVec.end() ) {
      if ( m_debug ) std::cout << "Configuring for a new Wheel" << wheel << std::endl;
      wheelmap = new RPCWheelMap ( wheel );
      m_wheelMapVec[wheel] = wheelmap;
      wheelmap->addHit( bx, sector, layer );
    }
    else {
      wheelmap = (*itr).second;
      wheelmap->addHit( bx, sector, layer );
    }
    
    if ( m_debug ) std::cout << "looping over digis 2 ..." << std::endl;
    
  }
      
  //... set up data to be processed
  int bx(0);
  int code(0);
  int bxsign(1);
  int wheel(-10);
  
  std::map<int, RPCWheelMap*>::iterator itr;
  itr = m_wheelMapVec.begin();
  
  while( itr != m_wheelMapVec.end()){
    
    (*itr).second->prepareData();
    
    wheel = (*itr).second->wheelIdx();
    
    for( int k = 0; k < m_maxBx; ++k ) {
      
      bx   = k - m_maxBxWindow;
      
      if ( bx != 0 ) bxsign = ( bx / abs(bx) );
      else bxsign = 1;
      code = bxsign * ( 1000000*abs(bx) + 10000*wheel );
      
      TTUInput * signal = & (*itr).second->m_ttuinVec[k];
      
      m_data.insert( std::make_pair( code , signal ) );
      
    }
    ++itr;
  }
  
  //...
  
  if ( m_data.size() <= 0 ) return 0;
  
  return 1;
  
}

int ProcessDigiGlobalSignal::getBarrelLayer( const int & layer, const int & station )
{
  
  //... Calculates the generic Barrel Layer (1 to 6)
  int blayer(0);
  
  if ( station < 3 ) {
    blayer = ( (station - 1) * 2 ) + layer;
  }
  else {
    blayer = station + 2;
  }
  
  return blayer;
  
}

void ProcessDigiGlobalSignal::reset()
{
  
  std::map<int, RPCWheelMap*>::iterator itr;
  
  for(itr = m_wheelMapVec.begin(); itr != m_wheelMapVec.begin(); ++itr)
    if( (*itr).second ) delete (*itr).second;
  
  m_wheelMapVec.clear();
  
}
