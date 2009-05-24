// $Id: RPCData.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RPCData
//
// 2008-11-18 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCData::RPCData() {
  
  m_wheel     = 10;
  m_sec1      = new int[6];
  m_sec2      = new int[6];
  m_orsignals = new RBCInput[6];
  
}
//=============================================================================
// Destructor
//=============================================================================
RPCData::~RPCData() {
  
  delete [] m_sec1;
  delete [] m_sec2;
  delete [] m_orsignals;
  
} 

//=============================================================================

std::istream& operator>>(std::istream &istr , RPCData & rhs) 
{
  
  (istr) >> rhs.m_wheel;
  for(int k=0; k < 6; ++k)
  {
    (istr) >> rhs.m_sec1[k] >> rhs.m_sec2[k];
    (istr) >> rhs.m_orsignals[k];
  }

  return istr;
    
}

std::ostream& operator<<(std::ostream& ostr , RPCData & rhs) 
{
  
  ostr << rhs.m_wheel << '\t';
  for(int k=0; k < 6; ++k)
  {
    ostr << rhs.m_sec1[k] << '\t' <<  rhs.m_sec2[k] << '\n';
    ostr << rhs.m_orsignals[k];
  }
  
  return ostr;
  
}


