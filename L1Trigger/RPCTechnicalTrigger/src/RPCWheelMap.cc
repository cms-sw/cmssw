// $Id: RPCWheelMap.cc,v 1.2 2009/05/08 10:24:05 aosorio Exp $
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

  //... considering that we have a bxing in the range [-3,+3]
  m_wheelid    = wheelid;
  m_wheelmap   = new std::bitset<6>[12];
  m_wheelmapbx = new std::bitset<6>[12 * 7];
  m_ttuin      = new TTUInput();
  
  for(int i=0; i < 12; ++i)
    m_wheelmap[i].reset();
  
  for(int i=0; i < 84; ++i)
    m_wheelmapbx[i].reset();

  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
RPCWheelMap::~RPCWheelMap() {
  
  if ( m_wheelmap ) delete[] m_wheelmap;
  if ( m_wheelmapbx ) delete[] m_wheelmapbx;
  if ( m_ttuin ) delete m_ttuin;
  
} 

//=============================================================================
void RPCWheelMap::addHit( int bx, int sec, int layer)
{
  
  int indx1 = bx + 3;
  int indx2 = sec + indx1*12;
  m_wheelmapbx[ indx2 ].set( layer-1, 1); 
  
}

void RPCWheelMap::contractMaps()
{
  
  std::bitset<6> tmp;
    
  for(int i=0; i < 12; ++i) {
    tmp.reset();
    for(int j=0; j < 7; ++j) {
      int indx = i + j*12;
      tmp |= m_wheelmapbx[ indx ];
    }
    m_wheelmap[i] = tmp;
  }
  
  if( m_debug ) {
    std::string test;
    for(int i=0; i<12; ++i) {
      test = m_wheelmap[i].to_string<char,std::char_traits<char>,std::allocator<char> >();
      std::cout << "sec: " << i << " " << test << std::endl;
    }
  }
  
}

void RPCWheelMap::prepareData()
{
  
  for(int i=0; i < 12; ++i) {
    m_ttuin->input_sec[i] = m_wheelmap[i]; 
  }
  
}
