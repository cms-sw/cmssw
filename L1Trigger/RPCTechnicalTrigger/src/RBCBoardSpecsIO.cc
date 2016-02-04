// $Id: RBCBoardSpecsIO.cc,v 1.1 2009/06/04 11:52:59 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCBoardSpecsIO.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCBoardSpecsIO
//
// 2008-12-16 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCBoardSpecsIO::RBCBoardSpecsIO( ) {

}
//=============================================================================
// Destructor
//=============================================================================
RBCBoardSpecsIO::~RBCBoardSpecsIO() {} 

//=============================================================================
std::istream& operator>>(std::istream & istr, RBCBoardSpecsIO::RBCBoardConfig & rhs)
{
  
  std::string logitype;
  
  istr >> rhs.m_Firmware      ;
  istr >> rhs.m_WheelId       ;
  istr >> rhs.m_Latency       ;
  istr >> rhs.m_MayorityLevel ;
  istr >> logitype ;

  //...m_MaskedOrInput is a vector of size 30
  for(int i=0; i < 30; ++i) {
    int mask(0);
    istr >> mask;
    rhs.m_MaskedOrInput.push_back(mask);
  }
  
  //...m_ForcedOrInput is a vector of size 30
  for(int i=0; i < 30; ++i) {
    int force(0);
    istr >> force;
    rhs.m_ForcedOrInput.push_back(force);
  }

  rhs.m_LogicType = logitype;
  
  return istr;
  
}

