// $Id: RBCLinkBoardGLSignal.cc,v 1.2 2009/05/16 19:43:32 aosorio Exp $
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
