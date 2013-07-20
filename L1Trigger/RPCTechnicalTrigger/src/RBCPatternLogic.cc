// $Id: RBCPatternLogic.cc,v 1.3 2009/06/07 21:18:50 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPatternLogic.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCPatternLogic
//
// 2008-10-15 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCPatternLogic::RBCPatternLogic(  ) {

}
//=============================================================================
// Destructor
//=============================================================================
RBCPatternLogic::~RBCPatternLogic() {} 

//=============================================================================

void RBCPatternLogic::process( const RBCInput & _input, std::bitset<2> & _decision ) 
{
  std::cout << "RBCPatternLogic> Working with pattern logic" << '\n';
  
  _decision.set(0,1);
  _decision.set(1,1);
  
  //...Layer information:
  for(int k=0; k < 6; ++k) {
    m_testlayer[0].set(k,1);
    m_testlayer[1].set(k,0);
  }
  

  //....


}

void RBCPatternLogic::setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & specs )
{
  
  
  
  
  
}
