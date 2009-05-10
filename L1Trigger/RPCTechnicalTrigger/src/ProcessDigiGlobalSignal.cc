// $Id: ProcessDigiGlobalSignal.cc,v 1.1 2009/05/08 10:24:05 aosorio Exp $
// Include files 


// local
#include "L1Trigger/RPCTechnicalTrigger/src/ProcessDigiGlobalSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUGlobalSignal.h" 

//-----------------------------------------------------------------------------
// Implementation file for class : ProcessDigiGlobalSignal
//
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

  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
ProcessDigiGlobalSignal::~ProcessDigiGlobalSignal() {
  
  std::vector<RPCWheelMap*>::iterator itr;
  for(itr = m_wheelmapvec.begin(); itr != m_wheelmapvec.begin(); ++itr)
    delete (*itr);
  m_wheelmapvec.clear();
  
  if( m_wmin ) delete m_wmin;
  
} 

//=============================================================================
int ProcessDigiGlobalSignal::next() 
{
  
  //. Generate a Wheel map from Digi collection
  int prev_wheel_id = -10;
  RPCWheelMap * wheelmap = NULL;
  
  for (m_detUnitItr = (*m_ptr_digiColl)->begin(); 
       m_detUnitItr != (*m_ptr_digiColl)->end(); ++m_detUnitItr ) {
    
    m_digiItr = (*m_detUnitItr ).second.first;
    int bx = (*m_digiItr).bx();
    
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
    
    
    if ( wheel != prev_wheel_id ) {
      wheelmap = new RPCWheelMap( wheel );
      m_wheelmapvec.push_back ( wheelmap );
    }
    
    wheelmap->addHit( bx, sector, layer );
    
    prev_wheel_id = wheel;
    
  }
  
  //.. add up all bunch X info into a single map
  
  std::vector<RPCWheelMap*>::const_iterator itr;
  itr = m_wheelmapvec.begin();
  while( itr!=m_wheelmapvec.end()){
    (*itr)->contractMaps();
    ++itr;
  }
  
  //... set up data to be processed
  
  int wheel      = -10;
  int prev_wheel = -100;
  itr = m_wheelmapvec.begin();
  while( itr!=m_wheelmapvec.end()){
    wheel = (*itr)->wheelid();
    if ( wheel != prev_wheel ) {
      (*itr)->prepareData();
      m_data.insert( std::make_pair( (*itr)->wheelid() * 10000 , (*itr)->m_ttuin ) );
    }
    prev_wheel = wheel;
    ++itr;
  }
  
  //...
  
  return 1;
  
}

int ProcessDigiGlobalSignal::getBarrelLayer( const int & _layer, const int & _station )
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

