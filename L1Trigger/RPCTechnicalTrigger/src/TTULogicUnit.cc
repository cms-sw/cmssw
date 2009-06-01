// $Id: TTULogicUnit.cc,v 1.2 2009/05/08 10:24:05 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogicUnit.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTULogicUnit
//
// 2008-10-25 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTULogicUnit::TTULogicUnit( ) : RPCLogicUnit () {
  
  m_logtool = new LogicTool<TTULogic>();
  m_debug = false;
  
}

TTULogicUnit::TTULogicUnit( const char * logic_type ) : RPCLogicUnit () {

  m_logtool = new LogicTool<TTULogic>();
  m_logtype = std::string( logic_type );
  m_debug = false;
  
}
//=============================================================================
// Destructor
//=============================================================================
TTULogicUnit::~TTULogicUnit() {
  
  if (m_logtool) {
    if ( m_logtool->endjob() )
      delete m_logtool;
  }

} 

//=============================================================================
bool TTULogicUnit::initialise() 
{
  
  bool status(true);
  
  status = m_logtool->initialise();
  if ( !status ) { 
    if( m_debug ) std::cout << "TTULogicUnit> Problem initialising LogicTool \n"; 
    return 0; };

  m_logic  = dynamic_cast<TTULogic*> ( m_logtool->retrieve(m_logtype) );
  
  if ( ! m_logic ) { 
    if( m_debug ) std::cout << "TTULogicUnit> No logic found \n"; 
    return 0; };
  
  return 1;
  
}

void TTULogicUnit::setlogic( const char * logic )
{
  m_logtype = std::string( logic );
}

void TTULogicUnit::run( const TTUInput & input )
{
 
  m_logic->process( input );
 
}
