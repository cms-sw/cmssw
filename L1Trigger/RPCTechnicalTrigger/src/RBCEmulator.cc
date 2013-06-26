// $Id: RBCEmulator.cc,v 1.11 2009/07/04 20:07:40 aosorio Exp $
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
  m_rbcinfo = new RBCId();
  m_input   = new RBCInput();
    
  m_layersignal[0] = new std::bitset<6>();
  m_layersignal[1] = new std::bitset<6>();
  m_layersignalVec.push_back( m_layersignal[0] );
  m_layersignalVec.push_back( m_layersignal[1] );

  m_debug   = false;
  
}

RBCEmulator::RBCEmulator( const char * logic_type ) {
  
  m_signal  = NULL;
  m_logtype = std::string( logic_type );
  m_rbcinfo = new RBCId();
  m_input   = new RBCInput();
  m_rbcconf = dynamic_cast<RBCConfiguration*> (new RBCBasicConfig(logic_type));
  
  m_layersignal[0] = new std::bitset<6>();
  m_layersignal[1] = new std::bitset<6>();
  m_layersignalVec.push_back( m_layersignal[0] );
  m_layersignalVec.push_back( m_layersignal[1] );
  
  m_debug   = false;
  
}

RBCEmulator::RBCEmulator( const char * f_name  , const char * logic_type ) {
  
  m_signal  = dynamic_cast<ProcessInputSignal*>(new RBCProcessTestSignal( f_name ));
  m_logtype = std::string( logic_type );
  m_rbcinfo = new RBCId();
  m_input   = new RBCInput();
  m_rbcconf = dynamic_cast<RBCConfiguration*> (new RBCBasicConfig(logic_type));

  m_layersignal[0] = new std::bitset<6>();
  m_layersignal[1] = new std::bitset<6>();
  m_layersignalVec.push_back( m_layersignal[0] );
  m_layersignalVec.push_back( m_layersignal[1] );
  
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

  std::vector<std::bitset<6>*>::iterator itr;
  for(itr =  m_layersignalVec.begin(); itr != m_layersignalVec.end(); ++itr)
    delete (*itr);
  
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

void RBCEmulator::setid( int wh, int * sec)
{
  m_rbcinfo->setid ( wh, sec);
}

void RBCEmulator::emulate() 
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting test emulation" << std::endl;
  
  std::bitset<2> decision;
  
  while ( m_signal->next() ) 
  {
    
    RPCInputSignal * data = m_signal->retrievedata();
    (*m_input) = * dynamic_cast<RBCLinkBoardSignal*>( data )->m_linkboardin ;
    
    m_rbcconf->m_rbclogic->run( (*m_input) , decision );
    
    m_layersignal[0] = m_rbcconf->m_rbclogic->getlayersignal( 0 );
    m_layersignal[1] = m_rbcconf->m_rbclogic->getlayersignal( 1 );
    
    printlayerinfo();
    
    if ( m_debug ) std::cout << decision[0] << " " << decision[1] << std::endl;
    
  }
  
  if( m_debug ) std::cout << "RBCEmulator> end test emulation" << std::endl;
  
}

void RBCEmulator::emulate( RBCInput * in )
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting emulation" << std::endl;
  
  std::bitset<2> decision;
  
  in->setWheelId( m_rbcinfo->wheel() );
  
  (*m_input) =  (*in);
  
  if( m_debug ) std::cout << "RBCEmulator> copied data" << std::endl;

  //.. mask and force as specified in hardware configuration
  m_rbcconf->preprocess( (*m_input) );

  if( m_debug ) std::cout << "RBCEmulator> preprocessing done" << std::endl;
    
  m_rbcconf->m_rbclogic->run( (*m_input) , decision );

  if( m_debug ) std::cout << "RBCEmulator> applying logic" << std::endl;
  
  m_layersignal[0] = m_rbcconf->m_rbclogic->getlayersignal( 0 );
  m_layersignal[1] = m_rbcconf->m_rbclogic->getlayersignal( 1 );

  m_decision.set(0, decision[0] );
  m_decision.set(1, decision[1] );
    
  if( m_debug ) {
    printlayerinfo();
    std::cout << decision[0] << " " << decision[1] << std::endl;
    std::cout << "RBCEmulator> end emulation" << std::endl;
  }
  
  decision.reset();  
  
}

void RBCEmulator::reset()
{

  m_decision.reset();
  m_layersignal[0]->reset();
  m_layersignal[1]->reset();
  
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
