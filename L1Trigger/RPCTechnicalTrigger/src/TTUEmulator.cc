// $Id: TTUEmulator.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $
// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/src/TTUEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUBasicConfig.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLinkBoardGLSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUGlobalSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUEmulator
//
// 2008-10-15 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUEmulator::TTUEmulator( int _id, int _mxw  ) 
{
  
  m_id        = _id;
  m_maxwheels = _mxw;
  int tmp[6]  = {1, -2, 0, 0, -1, 2};
  
  for( int k=0; k < 6; ++k) m_wheelids[k]=tmp[k];
  
  for( int k=0; k < m_maxwheels; ++k ) 
    m_Wheels[k] = new RPCWheel( m_wheelids[(_id*2)+(k-2)] );
  
  m_ttuin[0] = new TTUInput();
  m_ttuin[1] = new TTUInput();

  m_trigger.reset();
    
  m_mode = 1;
  
}


TTUEmulator::TTUEmulator( int _id, const char * rbclogic_type, const char * ttulogic_type, int _mxw  ) 
{
  
  m_id        = _id;
  m_maxwheels = _mxw;
  int tmp[6]  = {1, -2, 0, 0, -1, 2};
  
  for( int k=0; k < 6; ++k) m_wheelids[k]=tmp[k];
  
  for( int k=0; k < m_maxwheels; ++k ) 
    m_Wheels[k] = new RPCWheel( m_wheelids[(_id*2)+(k-2)], rbclogic_type );
  
  m_ttuin[0] = new TTUInput();
  m_ttuin[1] = new TTUInput();
  
  m_ttuconf   = dynamic_cast<TTUConfiguration*> (new TTUBasicConfig (ttulogic_type));

  m_trigger.reset();
  
  m_mode = 1;
  
}

TTUEmulator::TTUEmulator( int _id, const char * f_name, const char * rbclogic_type, 
                          const char * ttulogic_type, int _mxw  ) 
{
  
  m_id        = _id;
  m_maxwheels = _mxw;
  int tmp[6]  = {1, -2, 0, 0, -1, 2};
  
  for( int k=0; k < 6; ++k) m_wheelids[k]=tmp[k];
  
  for( int k=0; k < m_maxwheels; ++k ) 
    m_Wheels[k] = new RPCWheel( m_wheelids[(_id*2)+(k-2)], f_name, rbclogic_type );
  
  m_ttuin[0] = new TTUInput();
  m_ttuin[1] = new TTUInput();
  
  m_ttuconf   = dynamic_cast<TTUConfiguration*> (new TTUBasicConfig (ttulogic_type));

  m_trigger.reset();

  m_mode = 1;
  
}

//=============================================================================
// Destructor
//=============================================================================
TTUEmulator::~TTUEmulator() {

  if ( m_ttuconf ) delete m_ttuconf;
  
  if ( m_ttuin  ) 
    for (int k=0; k < 2; ++k) 
      delete m_ttuin[k];
  
  if ( m_Wheels ) 
    for (int k=0; k  < m_maxwheels; ++k )
      delete m_Wheels[k];
  
} 

//=============================================================================
void TTUEmulator::setSpecifications( const TTUBoardSpecs * ttuspecs, const RBCBoardSpecs * rbcspecs) 
{
  
  m_ttuconf   = dynamic_cast<TTUConfiguration*> (new TTUBasicConfig (ttuspecs));
  
  for( int k=0; k < m_maxwheels; ++k)
    m_Wheels[k]->setSpecifications( rbcspecs );

  std::vector<TTUBoardSpecs::TTUBoardConfig>::const_iterator itr;
  itr = m_ttuconf->m_ttuboardspecs->m_boardspecs.begin();
  
  m_mode = (*itr).m_triggerMode;
  
}

bool TTUEmulator::initialise()
{
  bool status(false);
  for( int k=0; k < m_maxwheels; ++k)
    status = m_Wheels[k]->initialise( );
  
  status = m_ttuconf->initialise();
  
  if ( !status ) { 
    std::cout << "TTUEmulator> Problem initialising the Configuration \n"; 
    return 0; };
  
  return status;
  
}

void TTUEmulator::emulate() 
{
  
  //... only for testing
  for( int k=0; k < m_maxwheels; ++k ) 
    m_Wheels[k]->emulate();
  
}

void TTUEmulator::processlocal( RPCInputSignal * signal ) 
{
  
  //. 
  bool trg(false); 

  m_trigger.reset();
    
  std::map<int,RBCInput*> * linkboardin;
  linkboardin = dynamic_cast<RBCLinkBoardGLSignal*>( signal )->m_linkboardin;
  
  for( int k=0; k < m_maxwheels; ++k )
  {
    
    if ( m_Wheels[k]->process( (*linkboardin) ) ) {
      
      m_Wheels[k]->createWheelMap();
      m_Wheels[k]->retrieveWheelMap( (*m_ttuin[k]) );
      
      //.. execute here the Tracking Algorithm or any other selected logic
      
      m_ttuconf->m_ttulogic->run( (*m_ttuin[k]) );
      
      //... and produce a Wheel level trigger
      trg = m_ttuconf->m_ttulogic->isTriggered();
      
      m_trigger.set(k,trg);
      
      std::cout << "TTUEmulator::processlocal ttuid: " << m_id 
                << " wheel: "       << m_Wheels[k]->getid()
                << " response: "    << trg << std::endl;
    }
    
  }
  
  std::cout << "TTUEmulator::processlocal> done with this TTU: " << m_id << std::endl;
  
}

void TTUEmulator::processglobal( RPCInputSignal * signal ) 
{
  
  //. 
  bool trg(false);

  m_trigger.reset();

  std::map<int,TTUInput*> * wheelmapin;
  wheelmapin = dynamic_cast<TTUGlobalSignal*>( signal )->m_wheelmap;
  
  for( int k=0; k < m_maxwheels; ++k )
  {
    
    if ( m_Wheels[k]->process( (*wheelmapin) ) ) {
      
      m_Wheels[k]->retrieveWheelMap( (*m_ttuin[k]) );
      
      //.. mask and force as specified in hardware configuration
      m_ttuconf->preprocess( (*m_ttuin[k]) );
      
      //.. execute here the Tracking Algorithm or any other selected logic
      
      m_ttuconf->m_ttulogic->run( (*m_ttuin[k]) );
      
      //... and produce a Wheel level trigger
      trg = m_ttuconf->m_ttulogic->isTriggered();
      
      m_trigger.set(k,trg);
      
      std::cout << "TTUEmulator::processglobal ttuid: " << m_id 
                << " wheel: "       << m_Wheels[k]->getid()
                << " response: "    << trg << std::endl;


    }
    
  }

  std::cout << "TTUEmulator::processglobal> done with this TTU: " << m_id << std::endl;
  
}


//.................................................................

void TTUEmulator::printinfo() 
{
  
  std::cout << "TTUEmulator: " << m_id << '\n';
  for( int k=0; k < m_maxwheels; ++k ) 
    m_Wheels[k]->printinfo();
  
}
