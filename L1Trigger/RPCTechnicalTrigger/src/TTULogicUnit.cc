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
  
  m_debug = false;
  
}

TTULogicUnit::TTULogicUnit( const char * logic_type ) : RPCLogicUnit () {

  m_logtype = std::string( logic_type );
  m_debug = false;
  
}

//=============================================================================
bool TTULogicUnit::initialise() 
{
  
  LogicTool<TTULogic> logtool;
  m_logic  = logtool.retrieve(m_logtype);
  
  if ( ! m_logic ) { 
    if( m_debug ) std::cout << "TTULogicUnit> No logic found \n"; 
    return false; };
  
  return true;
  
}

void TTULogicUnit::setlogic( const char * logic )
{

  m_logtype = std::string( logic );

}

void TTULogicUnit::setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & boardSpcs )
{
  
  m_logic->setBoardSpecs ( boardSpcs );
  
}

void TTULogicUnit::run( const TTUInput & input )
{

  //... check the thresholds

  //... by Sector
  
  //... by Tower

  //... by Wheel

  m_logic->process( input );

  //m_logic->m_triggersignal = false;
  
 
}

void TTULogicUnit::run( const TTUInput & input , int option )
{

  m_logic->setOption( option );
  m_logic->process( input );

  //m_logic->m_triggersignal = false;
  
}
