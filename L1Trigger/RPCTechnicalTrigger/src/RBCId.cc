// $Id: RBCId.cc,v 1.8 2009/08/09 11:11:37 aosorio Exp $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCId.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCId
//
// 2008-10-12 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCId::RBCId(  ) {
  m_wheel = -9;
  m_sector[0] = 100;
  m_sector[1] = 101;
}

RBCId::RBCId( int _w, int * _s ) 
{
  m_wheel     = _w;
  m_sector[0] = _s[0];
  m_sector[1] = _s[1];
}

RBCId::RBCId( const RBCId & id ) 
{
  m_wheel     = id.wheel();
  m_sector[0] = id.sector(0);
  m_sector[1] = id.sector(1);
}

//=============================================================================
// Destructor
//=============================================================================
RBCId::~RBCId() {} 

//=============================================================================
void RBCId::printinfo()
{
  
  std::cout << " ---->whe " << m_wheel << '\n';
  std::cout << " ---->sec " << m_sector[0] << '\t' << m_sector[1] << '\n';
  
}
