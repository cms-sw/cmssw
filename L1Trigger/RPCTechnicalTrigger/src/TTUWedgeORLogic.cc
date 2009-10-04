// $Id: TTUWedgeORLogic.cc,v 1.3 2009/09/15 14:42:10 aosorio Exp $
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
 
  m_wheelMajority[ boardspecs.m_Wheel1Id ] = 4;
  
  if ( (boardspecs.m_MaxNumWheels > 1) && (boardspecs.m_Wheel2Id != 0) )
    m_wheelMajority[ boardspecs.m_Wheel2Id ] = 3;

  if ( m_debug ) std::cout << "TTUWedgeORLogic::setBoardSpecs> intialization: " 
                           << m_wheelMajority.size() << '\t'
                           << boardspecs.m_MaxNumWheels << '\t'
                           << boardspecs.m_Wheel1Id << '\t'
                           << boardspecs.m_Wheel2Id << '\t'
                           << m_wheelMajority[ boardspecs.m_Wheel1Id ] << '\t'
                           << m_wheelMajority[ boardspecs.m_Wheel2Id ] << '\n';
    
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
  
  int nhits(0);
  int firstsector = m_wedgeSector[ (m_option-1) ] -1;
  
  for(int j = 0; j < m_maxsectors; ++j)  {
    nhits += inmap.input_sec[ firstsector + j ].count();
  }

  int majority = m_wheelMajority[ inmap.m_wheelId ];

  if ( m_debug ) std::cout << "TTUWedgeORLogic::setBoardSpecs> configuration W: " 
                           << inmap.m_wheelId << '\t' << "M: " << majority << '\n';
  
  if ( nhits >= majority) m_triggersignal = true;
  
  if( m_debug ) 
    std::cout << "TTUWedgeORLogic wedge decision:" 
              << m_option << '\t' 
              << firstsector << '\t' 
              << "nhits: " << nhits << '\t'
              << "Dec: "   << m_triggersignal << std::endl;
  
  if( m_debug ) std::cout << "TTUWedgeORLogic>process ends" << std::endl;
  
  return true;
  
}
