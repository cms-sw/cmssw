// $Id: RBCEmulator.cc,v 1.6 2009/05/08 10:24:05 aosorio Exp $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCBasicConfig.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCEmulator
//
// 2008-10-10 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCEmulator::RBCEmulator( ) {
  
  m_signal  = NULL;
  m_logtype = std::string("TestLogic");
  m_input   = new RBCInput();
  m_rbcinfo = new RBCId();
  
  m_debug   = false;
    
}

RBCEmulator::RBCEmulator( const char * logic_type ) {
  
  m_signal  = NULL;
  m_logtype = std::string( logic_type );
  m_input   = new RBCInput();
  m_rbcinfo = new RBCId();
  m_rbcconf = dynamic_cast<RBCConfiguration*> (new RBCBasicConfig(logic_type));
  
  m_debug   = false;

}

RBCEmulator::RBCEmulator( const char * f_name  , const char * logic_type ) {
  
  m_signal  = dynamic_cast<ProcessInputSignal*>(new RBCProcessTestSignal( f_name ));
  m_logtype = std::string( logic_type );
  m_input   = new RBCInput();
  m_rbcinfo = new RBCId();
  m_rbcconf = dynamic_cast<RBCConfiguration*> (new RBCBasicConfig(logic_type));

  m_debug   = false;
  
}

//=============================================================================
// Destructor
//=============================================================================
RBCEmulator::~RBCEmulator() {
  
  if (m_signal)  delete m_signal;
  if (m_rbcconf) delete m_rbcconf;
  if (m_rbcinfo) delete m_rbcinfo;
  if (m_input)   delete m_input;
  
} 

//=============================================================================
void RBCEmulator::setSpecifications( const RBCBoardSpecs * rbcspecs) 
{

  m_rbcconf = dynamic_cast<RBCConfiguration*> (new RBCBasicConfig(rbcspecs, m_rbcinfo));

}

bool RBCEmulator::initialise() 
{
  
  bool status(true);
  
  status = m_rbcconf->initialise();
  
  if ( !status ) { 
    if( m_debug ) std::cout << "RBCEmulator> Problem initialising the Configuration \n"; 
    return 0; };
  
  return 1;
  
}

void RBCEmulator::setid( int _wh, int * _sec)
{
  m_rbcinfo->setid ( _wh, _sec);
}

void RBCEmulator::emulate() 
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting test emulation" << std::endl;
  
  std::bitset<2> _decision;
  
  while ( m_signal->next() ) 
  {
    
    RPCInputSignal * data = m_signal->retrievedata();
    (*m_input) = * dynamic_cast<RBCLinkBoardSignal*>( data )->m_linkboardin ;
    
    m_rbcconf->m_rbclogic->run( (*m_input) , _decision );
    
    m_layersignal[0] = m_rbcconf->m_rbclogic->getlayersignal( 0 );
    m_layersignal[1] = m_rbcconf->m_rbclogic->getlayersignal( 1 );
    
    printlayerinfo();
    
    std::cout << _decision[0] << " " << _decision[1] << std::endl;
    
  }
  
  if( m_debug ) std::cout << "RBCEmulator> end test emulation" << std::endl;
  
}

void RBCEmulator::emulate( RBCInput * _in )
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting emulation" << std::endl;
  
  std::bitset<2> _decision;
  
  (*m_input) =  (*_in);

  //.. mask and force as specified in hardware configuration
  m_rbcconf->preprocess( (*m_input) );   
    
  m_rbcconf->m_rbclogic->run( (*m_input) , _decision );
  
  m_layersignal[0] = m_rbcconf->m_rbclogic->getlayersignal( 0 );
  m_layersignal[1] = m_rbcconf->m_rbclogic->getlayersignal( 1 );
  
  if( m_debug ) {
    printlayerinfo();
    std::cout << _decision[0] << " " << _decision[1] << std::endl;
    std::cout << "RBCEmulator> end emulation" << std::endl;
  }
    
}

void RBCEmulator::printinfo()
{
  
  if( m_debug ) {
    std::cout << "RBC --> \n";
    m_rbcinfo->printinfo();
  }
  
}

void RBCEmulator::printlayerinfo()
{

  std::cout << "Sector summary by layer: \n";
  for(int i=0; i < 6; ++i)
    std::cout << (*m_layersignal[0])[i] << '\t' 
              << (*m_layersignal[1])[i] << '\n';  
  
}
