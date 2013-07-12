// $Id: TTUInput.cc,v 1.6 2009/06/17 15:27:24 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Implementation file for class : TTUInput
//
// 2008-10-16 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUInput::TTUInput(  ) {

  m_bx = 0;
  m_wheelId = 0;
  m_hasHits = false;
  input_sec = new std::bitset<6>[12];
  m_rbcDecision.reset();
  
  for(int i=0; i < 12; ++i)
    input_sec[i].reset();
  
  m_debug = false;

}
//=============================================================================
// Destructor
//=============================================================================
TTUInput::~TTUInput() {

  m_hasHits = false;
  if ( input_sec ) delete[] input_sec;

} 
//=============================================================================

void TTUInput::reset() 
{
  
  m_bx = 0;
  m_wheelId = 0;
  m_hasHits = false;
  
  for(int i=0; i < 12; ++i)
    input_sec[i].reset();

  m_rbcDecision.reset();
    
}

void TTUInput::mask( const std::vector<int> & maskvec )
{
  
  //for(int i=0; i < 15; ++i) 
  //  if ( maskvec[i] ) input_sec[0].set(i,0);
  
  //for(int i=15; i < 30; ++i)
  //  if ( maskvec[i] ) input_sec[1].set( (i-15),0);
  
}

void TTUInput::force( const std::vector<int> & forcevec )
{
  
  //if( m_debug ) std::cout << forcevec.size() << std::endl;
  
  //std::bitset<15> tmp;
  
  //for(int i=0; i < 15; ++i)
  //  tmp.set(i,forcevec[i]);
  
  //... operate on the first sector
  //input_sec[0]|=tmp;
  //tmp.reset();
  
  //for(int i=15; i < 30; ++i)
  //  tmp.set( (i-15),forcevec[i]);
  
  //input_sec[1]|=tmp;
  
  //tmp.reset();
  
}

