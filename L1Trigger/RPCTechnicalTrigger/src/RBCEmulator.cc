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
RBCEmulator::RBCEmulator( ):
  m_rbcinfo{},
  m_signal{},
  m_input{},
  m_logtype{"TestLogic"},
  m_debug{false}
 {
  m_layersignal[0] = &m_layersignalVec[0];
  m_layersignal[1] = &m_layersignalVec[1];
}

RBCEmulator::RBCEmulator( const char * logic_type ):
  m_rbcinfo{},
  m_signal{},
  m_rbcconf{std::make_unique<RBCBasicConfig>(logic_type)},
  m_input{},
  m_logtype{logic_type},
  m_debug{false}
 {
  m_layersignal[0] = &m_layersignalVec[0];
  m_layersignal[1] = &m_layersignalVec[1];
}

RBCEmulator::RBCEmulator( const char * f_name  , const char * logic_type ):
  m_rbcinfo{},
  m_signal{std::make_unique<RBCProcessTestSignal>( f_name )},
  m_rbcconf{std::make_unique<RBCBasicConfig>(logic_type)},
  m_input{},
  m_logtype{ logic_type },
  m_debug{false}
 {
  m_layersignal[0] = &m_layersignalVec[0];
  m_layersignal[1] = &m_layersignalVec[1];
}

//=============================================================================
void RBCEmulator::setSpecifications( const RBCBoardSpecs * rbcspecs) 
{

  m_rbcconf = std::make_unique<RBCBasicConfig>(rbcspecs, &m_rbcinfo);

}

bool RBCEmulator::initialise() 
{
  
  bool status(true);
  
  status = m_rbcconf->initialise();
  
  if ( !status ) { 
    if( m_debug ) std::cout << "RBCEmulator> Problem initialising the Configuration \n"; 
    return false; };
  
  return true;
  
}

void RBCEmulator::setid( int wh, int * sec)
{
  m_rbcinfo.setid ( wh, sec);
}

void RBCEmulator::emulate() 
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting test emulation" << std::endl;
  
  std::bitset<2> decision;
  
  while ( m_signal->next() ) 
  {
    
    RPCInputSignal * data = m_signal->retrievedata();
    m_input = dynamic_cast<RBCLinkBoardSignal*>( data )->m_linkboardin ;
    
    m_rbcconf->rbclogic()->run( m_input , decision );
    
    m_layersignal[0] = m_rbcconf->rbclogic()->getlayersignal( 0 );
    m_layersignal[1] = m_rbcconf->rbclogic()->getlayersignal( 1 );
    
    printlayerinfo();
    
    if ( m_debug ) std::cout << decision[0] << " " << decision[1] << std::endl;
    
  }
  
  if( m_debug ) std::cout << "RBCEmulator> end test emulation" << std::endl;
  
}

void RBCEmulator::emulate( RBCInput * in )
{
  
  if( m_debug ) std::cout << "RBCEmulator> starting emulation" << std::endl;
  
  std::bitset<2> decision;
  
  in->setWheelId( m_rbcinfo.wheel() );
  
  m_input =  (*in);
  
  if( m_debug ) std::cout << "RBCEmulator> copied data" << std::endl;

  //.. mask and force as specified in hardware configuration
  m_rbcconf->preprocess( m_input );

  if( m_debug ) std::cout << "RBCEmulator> preprocessing done" << std::endl;
    
  m_rbcconf->rbclogic()->run( m_input , decision );

  if( m_debug ) std::cout << "RBCEmulator> applying logic" << std::endl;
  
  m_layersignal[0] = m_rbcconf->rbclogic()->getlayersignal( 0 );
  m_layersignal[1] = m_rbcconf->rbclogic()->getlayersignal( 1 );

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

void RBCEmulator::printinfo() const
{
  
  if( m_debug ) {
    std::cout << "RBC --> \n";
    m_rbcinfo.printinfo();
  }
  
}

void RBCEmulator::printlayerinfo() const
{

  std::cout << "Sector summary by layer: \n";
  for(int i=0; i < 6; ++i)
    std::cout << (*m_layersignal[0])[i] << '\t' 
              << (*m_layersignal[1])[i] << '\n';  
  
}
