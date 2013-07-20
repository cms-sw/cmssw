// $Id: RPCLogicUnit.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCLogicUnit.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RPCLogicUnit
// Test
// 2008-10-25 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCLogicUnit::RPCLogicUnit( int _a, int _b, int _c ) {
  
  m_propA = _a;
  m_propB = _b;
  m_propC = _c;
  
}

//=============================================================================
// Destructor
//=============================================================================
RPCLogicUnit::~RPCLogicUnit() {
  
  
} 

//=============================================================================
