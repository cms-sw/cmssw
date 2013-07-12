// $Id: RBCLinkBoardGLSignal.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCLinkBoardGLSignal
//
// 2008-11-28 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCLinkBoardGLSignal::RBCLinkBoardGLSignal( std::map< int, RBCInput* >  * in ) {
  
  m_linkboardin = in;
  
}
//=============================================================================
// Destructor
//=============================================================================
RBCLinkBoardGLSignal::~RBCLinkBoardGLSignal() {

  m_linkboardin = NULL;
    
} 

//=============================================================================
