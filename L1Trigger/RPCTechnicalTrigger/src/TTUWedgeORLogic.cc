// $Id: TTUWedgeORLogic.cc,v 1.7 2012/02/09 13:00:01 eulisse Exp $
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
  
  //m_maxsectors = 3; //this is the size of the wedge

  //the key is the starting sector: sec 2 -> quadrant 7,8,9 and so on
 
  m_wedgeSector[2]   = 1; //this is the size of the wedge: sec 2
  m_wedgeSector[3]   = 1; //this is the size of the wedge: sec 3
  m_wedgeSector[4]   = 1; //this is the size of the wedge: sec 4
  m_wedgeSector[5]   = 1; //this is the size of the wedge: sec 5
  m_wedgeSector[6]   = 1; //this is the size of the wedge: sec 6
  m_wedgeSector[7]   = 3; //this is the size of the wedge: bottom quadrant 1
  m_wedgeSector[8]   = 3; //this is the size of the wedge: bottom quadrant 2
  m_wedgeSector[9]   = 3; //this is the size of the wedge: bottom quadrant 3
  m_wedgeSector[10]  = 3; //this is the size of the wedge: bottom quadrant 4
  m_wedgeSector[11]  = 3; //this is the size of the wedge: bottom quadrant 5

  //m_wedgeSector.push_back(2); //this is the starting sector for each wedge
  //m_wedgeSector.push_back(4);
  //m_wedgeSector.push_back(8);
  //m_wedgeSector.push_back(10);

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
 
  m_wheelMajority[ boardspecs.m_Wheel1Id ] = 3;
  
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
  
  // October 15 2009: A.Osorio
  // In this context m_option is the Selected Wedge/Quadrant (1,2,3,4...)
  // initially we had 4 quadrants
  // 1=*2-3-4 ; 2=*4-5-6; 3=*8-9-10; 4=*10-11-12
  // Now: we have 5 top sectors: 2,3,4,5,6 and 5 bottom quadrants +/-1 of the opposite sector
  
  int nhits(0);
  int sector_indx(0);
  int firstsector = m_option;
  
  m_maxsectors = m_wedgeSector[ firstsector ];
  
  for(int j = 0; j < m_maxsectors; ++j)  {
    sector_indx = (firstsector-1) + j;
    if( sector_indx >= 12 ) sector_indx = 0;
    nhits += inmap.input_sec[ sector_indx ].count();
  }
  
  //...introduce force logic
  bool use_forcing = false;
  
  if ( use_forcing ) {

    for(int j = 0; j < m_maxsectors; ++j)  {
      
      sector_indx = (firstsector-1) + j;
      
      if( firstsector <= 6 ) { //...only top sectors
        
        bool hasLayer1 = inmap.input_sec[sector_indx][0]; //layer 1: RB1in
        
        if ( ! hasLayer1 ) {
          m_triggersignal = false;
          return true;
        }
        
      }
      
    }
    
  }
  

  int majority = m_wheelMajority[ inmap.m_wheelId ];

  if ( m_debug ) std::cout << "TTUWedgeORLogic::setBoardSpecs> configuration W: " 
                           << inmap.m_wheelId << '\t' << "M: " << majority << '\n';
  
  if ( nhits >= majority) m_triggersignal = true;
  
  if( m_debug ) 
    std::cout << "TTUWedgeORLogic wedge decision: "
              << "wheel: "    << inmap.m_wheelId << '\t'
              << "quadrant: " << m_option << '\t' 
              << "fsector: "  << firstsector << '\t' 
              << "nhits: "    << nhits << '\t'
              << "maj: "      << majority << '\t'
              << "Dec: "      << m_triggersignal << std::endl;
  
  if( m_debug ) std::cout << "TTUWedgeORLogic>process ends" << std::endl;
  
  return true;
  
}
