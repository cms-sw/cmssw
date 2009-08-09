// $Id: TTUSectorORLogic.cc,v 1.1 2009/06/17 15:27:24 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUSectorORLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUSectorORLogic
//
// 2009-06-10 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUSectorORLogic::TTUSectorORLogic(  ) {

  m_triggersignal = false;

  m_maxsectors = 12;
  
  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
TTUSectorORLogic::~TTUSectorORLogic() {} 

//=============================================================================
void TTUSectorORLogic::setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & boardspecs ) 
{
  
  
}

bool TTUSectorORLogic::process( const TTUInput & inmap )
{
  
  if( m_debug) std::cout << "TTUSectorORLogic::process starts" << std::endl;
  
  m_triggersignal = false;
  
  for(int i=0; i < m_maxsectors; ++i) 
    m_triggersignal |= inmap.m_rbcDecision[i];
  
  if( m_debug ) 
    std::cout << "TTUSectorORLogic " << m_triggersignal << std::endl;
  
  if( m_debug ) std::cout << "TTUSectorORLogic>process ends" << std::endl;
  
  return true;
  
}

