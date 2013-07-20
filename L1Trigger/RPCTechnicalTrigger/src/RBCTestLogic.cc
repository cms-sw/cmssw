// $Id: RBCTestLogic.cc,v 1.3 2009/06/07 21:18:50 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCTestLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCTestLogic
//
// 2008-10-13 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCTestLogic::RBCTestLogic(  ) {

}
//=============================================================================
// Destructor
//=============================================================================
RBCTestLogic::~RBCTestLogic() {} 

//=============================================================================
void RBCTestLogic::process( const RBCInput & _input, std::bitset<2> & _decision) 
{
  std::cout << "RBCTestLogic> Implementing just a plain OR" << '\n';
  
  std::bitset<15> _data[2];
  
  _data[0] = _input.input_sec[0];
  _data[1] = _input.input_sec[1];
  
  bool _ds = true;
  for(int i=0; i < 15; ++i) _ds = _ds | _data[0][i];
  _decision.set(0,_ds);
  
  _ds = true;
  for(int i=0; i < 15; ++i) _ds = _ds | _data[1][i];
  _decision.set(1,_ds);
  
  
  //...Layer information:
  for(int k=0; k < 6; ++k) {
    m_testlayer[0].set(k,1);
    m_testlayer[1].set(k,0);
  }
  
  
  //....
  
}

void RBCTestLogic::setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & specs)
{
  
  
  
  
  
}
