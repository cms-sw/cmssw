// $Id: RPCWheelMap.cc,v 1.7 2009/06/07 21:18:51 aosorio Exp $
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
  
  m_maxBx = 7;
  m_maxBxWindow = 3; //... considering that we have a bxing in the range [-3,+3]
  m_maxSectors = 12;
  
  int maxMaps = m_maxBx * m_maxSectors;
  
  m_wheelid    = wheelid;
  m_wheelMap   = new std::bitset<6>[m_maxSectors];
  m_wheelMapBx = new std::bitset<6>[m_maxSectors * m_maxBx];
  m_ttuinVec   = new TTUInput[m_maxBx];
    
  for(int i=0; i < m_maxSectors; ++i)
    m_wheelMap[i].reset();
  
  for(int i=0; i < maxMaps; ++i)
    m_wheelMapBx[i].reset();
  
  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
RPCWheelMap::~RPCWheelMap() {
  
  if ( m_wheelMap )   delete[] m_wheelMap;
  if ( m_wheelMapBx ) delete[] m_wheelMapBx;
  if ( m_ttuinVec )   delete[] m_ttuinVec;
  
} 

//=============================================================================
void RPCWheelMap::addHit( int bx, int sec, int layer)
{
  
  // |--12--|--12--|--12--|--12--|--12--|--12--|--12--| (12 sectors x 6 layers x  7 bx)
  // 0.....11
  int indx1 = bx + m_maxBxWindow;
  int indx2 = sec + indx1*m_maxSectors;
  m_wheelMapBx[ indx2 ].set( layer-1, 1);
  
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


