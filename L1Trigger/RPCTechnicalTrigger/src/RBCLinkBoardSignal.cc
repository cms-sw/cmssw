// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCLinkBoardSignal
//
// 2008-11-27 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCLinkBoardSignal::RBCLinkBoardSignal( RBCInput * in ) {

  RBCInput * m_linkboardin = new RBCInput();
  (*m_linkboardin) = (*in);
  
}
//=============================================================================
// Destructor
//=============================================================================
RBCLinkBoardSignal::~RBCLinkBoardSignal() {
  
  if ( m_linkboardin ) delete m_linkboardin;

} 

//=============================================================================
