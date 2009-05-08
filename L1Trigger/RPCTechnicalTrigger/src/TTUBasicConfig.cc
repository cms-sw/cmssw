// $Id: TTUBasicConfig.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/src/TTUBasicConfig.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUBasicConfig
//
// 2008-10-31 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUBasicConfig::TTUBasicConfig( const TTUBoardSpecs * ttuspecs ) {

  m_ttuboardspecs = ttuspecs;
  m_ttulogic      = new TTULogicUnit();

  m_debug = false;
    
}

TTUBasicConfig::TTUBasicConfig( const char * _logic  ) {

  m_ttulogic = new TTULogicUnit( _logic );

  m_debug = false;
    
}

//=============================================================================
// Destructor
//=============================================================================
TTUBasicConfig::~TTUBasicConfig() {

  if (m_ttulogic) delete m_ttulogic;

  m_vecmask.clear();
  m_vecforce.clear();

} 

//=============================================================================
bool TTUBasicConfig::initialise()
{
  
  bool status(false);
  
  //.  read specifications
  
  std::vector<TTUBoardSpecs::TTUBoardConfig>::const_iterator itr;
  itr = m_ttuboardspecs->m_boardspecs.begin();
  
  // initialise logic unit
  m_ttulogic->setlogic( (*itr).m_LogicType.c_str() );
  
  status = m_ttulogic->initialise();

  // get mask and force vectors
  
  m_vecmask.assign( (*itr).m_MaskedSectors.begin(), (*itr).m_MaskedSectors.end() );
  m_vecforce.assign( (*itr).m_ForcedSectors.begin(), (*itr).m_ForcedSectors.end() );
  
  if ( !status ) { 
    if( m_debug ) std::cout << "TTUConfiguration> Problem initialising the logic unit\n"; 
    return 0; };
  
  return status;
  
}

void TTUBasicConfig::preprocess( TTUInput & input )
{
  
  if( m_debug ) std::cout << "TTUBasicConfig::preprocess> starts here" << std::endl;
  
  input.mask( m_vecmask );
  //input.force( m_vecforce );
  
  if( m_debug ) std::cout << "TTUBasicConfig::preprocess> done" << std::endl;
  
}
