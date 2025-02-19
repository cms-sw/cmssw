// $Id: TTUBoardSpecsIO.cc,v 1.1 2009/06/04 11:52:59 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUBoardSpecsIO.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUBoardSpecsIO
//
// 2008-12-16 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUBoardSpecsIO::TTUBoardSpecsIO(  ) {

}
//=============================================================================
// Destructor
//=============================================================================
TTUBoardSpecsIO::~TTUBoardSpecsIO() {} 

//=============================================================================
std::istream& operator>>(std::istream &istr, TTUBoardSpecsIO::TTUBoardConfig &rhs)
{
  
  std::string logitype;
  
  istr >> rhs.m_runId         ;
  istr >> rhs.m_runType       ;
  istr >> rhs.m_triggerMode   ;
  istr >> rhs.m_Firmware      ;
  istr >> rhs.m_LengthOfFiber ;
  istr >> rhs.m_Delay         ;
  istr >> rhs.m_MaxNumWheels  ;
  istr >> rhs.m_Wheel1Id      ;
  istr >> rhs.m_Wheel2Id      ;
  istr >> logitype            ;
  istr >> rhs.m_TrackLength   ;
  
  //...m_MaskedSectors is a vector of size 12
  for(int i=0; i < 12; ++i) {
    int mask(0);
    istr >> mask;
    rhs.m_MaskedSectors.push_back(mask);
  }
  
  //...m_ForcedSectors is a vector of size 12
  for(int i=0; i < 12; ++i) {
    int force(0);
    istr >> force;
    rhs.m_ForcedSectors.push_back(force);
  }
  
  rhs.m_LogicType = logitype;
  
  return istr;
  
}
