// $Id: TTUGlobalSignal.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUGlobalSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUGlobalSignal
//
// 2008-11-29 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUGlobalSignal::TTUGlobalSignal( std::map< int, TTUInput* >  * in ) {

  m_wheelmap = in; 
  
}
//=============================================================================
// Destructor
//=============================================================================
TTUGlobalSignal::~TTUGlobalSignal() {
  
  m_wheelmap = NULL;
  
} 

//=============================================================================
