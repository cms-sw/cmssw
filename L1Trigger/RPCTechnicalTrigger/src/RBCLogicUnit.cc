// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLogicUnit.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCLogicUnit
//
// 2008-10-25 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCLogicUnit::RBCLogicUnit( ) : RPCLogicUnit(),
                                m_debug{false}
{
}

RBCLogicUnit::RBCLogicUnit( const char * logic_type ) : RPCLogicUnit(),
  m_logtype{ logic_type },
  m_debug{ false }
{
}
//=============================================================================
// Destructor
//=============================================================================
RBCLogicUnit::~RBCLogicUnit() {
} 

//=============================================================================
bool RBCLogicUnit::initialise() 
{
  
  LogicTool<RBCLogic> logtool;

  m_logic  = logtool.retrieve(m_logtype);
  
  if ( ! m_logic ) { 
    if( m_debug ) std::cout << "RBCLogicUnit> No logic found \n"; 
    return false; };
  
  return true;
  
}

void RBCLogicUnit::setlogic( const char * _logic )
{
  m_logtype = std::string(_logic);
}

void RBCLogicUnit::setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & specs)
{

  m_logic->setBoardSpecs( specs );
  
}

void RBCLogicUnit::run( const RBCInput & _input , std::bitset<2> & _decision )
{
  
  m_logic->process( _input , _decision );
  m_layersignal[0] = m_logic->getlayersignal( 0 );
  m_layersignal[1] = m_logic->getlayersignal( 1 );
}
