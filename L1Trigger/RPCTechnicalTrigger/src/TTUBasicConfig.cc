// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUBasicConfig.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUBasicConfig
//
// 2008-10-31 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUBasicConfig::TTUBasicConfig( const TTUBoardSpecs * ttuspecs ):
  TTUConfiguration(ttuspecs) ,
  m_debug{false}
{
}

TTUBasicConfig::TTUBasicConfig( const char * logic  ):
  TTUConfiguration(logic),
  m_debug{false}
{
}

//=============================================================================
// Destructor
//=============================================================================
TTUBasicConfig::~TTUBasicConfig() {
} 

//=============================================================================
bool TTUBasicConfig::initialise( int line , int ttuid )
{
  
  bool status(false);
  
  //.  read specifications and set it to the corresponding TTU board
  
  std::vector<TTUBoardSpecs::TTUBoardConfig>::const_iterator itr;
  itr = m_ttuboardspecs->m_boardspecs.begin();
  
  int pos(0);
  int maxboards = m_ttuboardspecs->m_boardspecs.size();
  
  for( pos=0; pos < maxboards; ++pos) {
    if ( m_debug ) std::cout << "TTUBasicConfig::initialise> " 
                             << m_ttuboardspecs->m_boardspecs[pos].m_Wheel1Id 
                             << std::endl;
    if ( m_ttuboardspecs->m_boardspecs[pos].m_runId == ttuid ) break;
    
  }
  
  // initialise logic unit
  
  if ( line == 2 ) {
    ttulogic()->setlogic( "WedgeORLogic" );
  } else {
    ttulogic()->setlogic( (*itr).m_LogicType.c_str() );
  }
  
  status = ttulogic()->initialise();
  
  //itr = m_ttuboardspecs->m_boardspecs.begin();
  
  ttulogic()->setBoardSpecs( m_ttuboardspecs->m_boardspecs[pos] );
  
  // get mask and force vectors
  
  m_vecmask.assign( (*itr).m_MaskedSectors.begin(), (*itr).m_MaskedSectors.end() );
  
  m_vecforce.assign( (*itr).m_ForcedSectors.begin(), (*itr).m_ForcedSectors.end() );
  
  if ( !status ) { 
    if( m_debug ) std::cout << "TTUConfiguration> Problem initialising the logic unit\n"; 
    return false; };
  
  return status;
  
}

void TTUBasicConfig::preprocess( TTUInput & input )
{
  
  if( m_debug ) std::cout << "TTUBasicConfig::preprocess> starts here" << std::endl;
  
  input.mask( m_vecmask );
  //input.force( m_vecforce );
  
  if( m_debug ) std::cout << "TTUBasicConfig::preprocess> done" << std::endl;
  
}
