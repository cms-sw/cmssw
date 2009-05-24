// $Id: TTUInput.cc,v 1.3 2009/05/10 00:33:18 aosorio Exp $
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
  
  input_sec = new std::bitset<6>[12];
  
  for(int i=0; i < 12; ++i)
    input_sec[i].reset();
  
  m_debug = false;

}
//=============================================================================
// Destructor
//=============================================================================
TTUInput::~TTUInput() {
  
  if ( input_sec ) delete[] input_sec;

} 
//=============================================================================

void TTUInput::reset() 
{
  
  for(int i=0; i < 12; ++i)
    input_sec[i].reset();
  
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

