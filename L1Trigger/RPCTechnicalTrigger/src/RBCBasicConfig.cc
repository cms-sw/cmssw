// $Id: RBCBasicConfig.cc,v 1.3 2009/05/16 19:43:31 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCBasicConfig.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCBasicConfig
//
// 2008-10-31 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCBasicConfig::RBCBasicConfig( const RBCBoardSpecs * rbcspecs , RBCId * info )
{
  
  m_rbcboardspecs  = rbcspecs;
  m_rbclogic       = new RBCLogicUnit();
  m_rbcinfo        = new RBCId( *info );

  m_debug = false;
    
}

RBCBasicConfig::RBCBasicConfig( const char * _logic ) {
  
  m_rbclogic  = new RBCLogicUnit( _logic );
  
}
//=============================================================================
// Destructor
//=============================================================================
RBCBasicConfig::~RBCBasicConfig() {
  
  if ( m_rbcinfo  ) delete m_rbcinfo;
  if ( m_rbclogic ) delete m_rbclogic;

  m_vecmask.clear();
  m_vecforce.clear();
  
}

//=============================================================================
bool RBCBasicConfig::initialise() 
{
  
  bool status(false);
  
  //.  read specifications
  
  std::vector<RBCBoardSpecs::RBCBoardConfig>::const_iterator itr;
  itr = m_rbcboardspecs->v_boardspecs.begin();
  
  // initialise logic unit
  m_rbclogic->setlogic( (*itr).m_LogicType.c_str() );
  status = m_rbclogic->initialise();
  
  m_rbclogic->setBoardSpecs( (*itr) );
  
  // get mask and force vectors
  
  m_vecmask.assign( (*itr).m_MaskedOrInput.begin(), (*itr).m_MaskedOrInput.end() );
  m_vecforce.assign( (*itr).m_ForcedOrInput.begin(), (*itr).m_ForcedOrInput.end() );
  
  if ( !status ) { 
    if( m_debug ) std::cout << "RBCConfiguration> Problem initialising the logic unit\n"; 
    return 0; };
  
  return 1;
  
}

void RBCBasicConfig::preprocess( RBCInput & input )
{
  
  if( m_debug ) std::cout << "RBCBasicConfig::preprocess> starts here" << std::endl;

  input.mask( m_vecmask );
  input.force( m_vecforce );
  
  if( m_debug ) std::cout << "RBCBasicConfig::preprocess> done" << std::endl;
  
}
