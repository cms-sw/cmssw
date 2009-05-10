// $Id: TTUInput.cc,v 1.2 2009/05/08 10:24:05 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUInput
//
// 2008-10-16 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUInput::TTUInput(  ) {
  
  input_sec = new std::bitset<6>[12];
  
  for(int i=0; i < 12; ++i) {
    for (int j=0; j < 6; ++j ) {
      input_sec[i].set(j,false);
    }
  }

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
  for(int i=0; i < 12; ++i) {
    for (int j=0; j < 6; ++j ) {
      input_sec[i].set(j,false);
    }
  }
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

