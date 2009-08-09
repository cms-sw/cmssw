// $Id: $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUWedgeORLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUWedgeORLogic
//
// 2009-08-09 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUWedgeORLogic::TTUWedgeORLogic(  ) {

  m_triggersignal = false;
  
  m_maxsectors = 3; //this is the size of the wedge

  m_wedgeSector.push_back(2); //this is the starting sector for each wedge
  m_wedgeSector.push_back(4);
  m_wedgeSector.push_back(8);
  m_wedgeSector.push_back(10);

  m_maxwedges = m_wedgeSector.size();
  
  m_option = 0;
  
  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
TTUWedgeORLogic::~TTUWedgeORLogic() {} 

//=============================================================================
void TTUWedgeORLogic::setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & boardspecs ) 
{
  
  
}

bool TTUWedgeORLogic::process( const TTUInput & inmap )
{
  
  if( m_debug) std::cout << "TTUWedgeORLogic::process starts" << std::endl;
  
  m_triggersignal = false;

  //
  //In this context m_option is the Selected Wedge/Quadrant (1,2,3,4...)
  // for the moment we have 4 quadrants
  // 1=*2-3-4 ; 2=*4-5-6; 3=*8-9-10; 4=*10-11-12
  //

  int firstsector = m_wedgeSector[ m_option ] -1;
  
  for(int j = 0; j < m_maxsectors; ++j)  {
    m_triggersignal |= inmap.m_rbcDecision[ firstsector + j ];
  }
  
  if( m_debug ) 
    std::cout << "TTUWedgeORLogic wedge decision:" << m_triggersignal << std::endl;

  if( m_debug ) std::cout << "TTUWedgeORLogic>process ends" << std::endl;
  
  return true;
  
}
