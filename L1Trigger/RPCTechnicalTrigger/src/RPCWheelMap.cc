// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCWheelMap.h"
#include <string>
#include <iostream>
//-----------------------------------------------------------------------------
// Implementation file for class : RPCWheelMap
//
// 2008-11-24 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCWheelMap::RPCWheelMap( int wheelid ) {
  
  m_wheelid    = wheelid;
  m_debug = false;
  
}

//=============================================================================
void RPCWheelMap::addHit( int bx, int sec, int layer)
{
  
  // |--12--|--12--|--12--|--12--|--12--|--12--|--12--| (12 sectors x 6 layers x  7 bx)
  // 0.....11
  int indx1 = bx + m_maxBxWindow;
  int indx2 = sec + indx1*m_maxSectors;
  m_wheelMapBx[ indx2 ].set( layer-1, true);
  
}

void RPCWheelMap::prepareData()
{
  
  bool anyHits(false);
  
  for(int bx=0; bx < m_maxBx; ++bx) {
  
    anyHits = false;
      
    for(int i=0; i < m_maxSectors; ++i) {
      
      int indx = i + bx*m_maxSectors;
      
      m_ttuinVec[bx].m_bx = ( bx - m_maxBxWindow );
      m_wheelMap[i] = m_wheelMapBx[ indx ];
      m_ttuinVec[bx].input_sec[i] = m_wheelMap[i]; 
    
      anyHits |= m_wheelMap[i].any();
      
      if( m_debug ) {
        std::string test;
        test = m_wheelMap[i].to_string<char,std::char_traits<char>,std::allocator<char> >();
        std::cout << "prepareData> sec: " << i << " " << test << " anyHits " << anyHits << std::endl;
      }
      
    }
    
    m_ttuinVec[bx].m_hasHits = anyHits;
    
  }
  
  if( m_debug ) std::cout << "prepareData> done." << '\n';
  
}


