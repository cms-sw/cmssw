// $Id: TTUPointingLogic.cc,v 1.1 2009/08/09 11:11:37 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUPointingLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUPointingLogic
//
// 2009-07-29 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUPointingLogic::TTUPointingLogic(  ) {

  m_triggersignal = false;
 
  m_debug = false;
  
    
}
//=============================================================================
// Destructor
//=============================================================================
TTUPointingLogic::~TTUPointingLogic() { 

} 

//=============================================================================
void TTUPointingLogic::setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & boardspecs ) 
{
  
  m_wedgeLogic->setBoardSpecs( boardspecs );
  
}

bool TTUPointingLogic::process( const TTUInput & inmap )
{
  
  if( m_debug) std::cout << "TTUPointingLogic::process starts" << std::endl;
  
  m_triggersignal = false;
  
  //m_ttuLogic->process( inmap );
  
  if( m_debug ) std::cout << "TTUPointingLogic>process ends" << std::endl;
  
  return true;
  
}

