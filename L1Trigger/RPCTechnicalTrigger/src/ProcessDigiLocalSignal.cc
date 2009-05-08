// $Id: $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/src/ProcessDigiLocalSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : ProcessDigiLocalSignal
//
// 2009-04-15 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
ProcessDigiLocalSignal::ProcessDigiLocalSignal(  const edm::ESHandle<RPCGeometry> & rpcGeom, 
                                                 const edm::Handle<RPCDigiCollection> & digiColl ) 
{
  
  m_ptr_rpcGeom  = & rpcGeom;
  m_ptr_digiColl = & digiColl;
  
  m_debug = false;
  
}

//=============================================================================
// Destructor
//=============================================================================
ProcessDigiLocalSignal::~ProcessDigiLocalSignal() {
  

  
} 

//=============================================================================
int ProcessDigiLocalSignal::next() {
  
  

  
  return 1;
  
}

