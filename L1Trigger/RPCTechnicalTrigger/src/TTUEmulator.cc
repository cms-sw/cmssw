// Include files 


#include <cmath>
#include <algorithm>
// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUBasicConfig.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUGlobalSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUEmulator
//
// 2008-10-15 : Andres Osorio
//-----------------------------------------------------------------------------

namespace {
  constexpr std::array<int, 6> wheelIds = { {1,  2, 0, 0, -1, -2} };
}

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUEmulator::TTUEmulator( int id, int mxw  ):
  m_maxWheels{mxw},
  m_id{id},
  m_mode{1},
  m_line{1},
  m_debug{ false }
{
  for( int k=0; k < m_maxWheels; ++k ) 
    m_Wheels[k].setProperties( wheelIds[(id*2)+(k-2)] );
}


TTUEmulator::TTUEmulator( int id, const char * rbclogic_type, const char * ttulogic_type, int mxw  ):
  m_maxWheels{mxw},
  m_id{id},
  m_mode{1},
  m_line{1},
  m_ttuconf{std::make_unique<TTUBasicConfig>(ttulogic_type)},
  m_debug{false}
{
  for( int k=0; k < m_maxWheels; ++k ) 
    m_Wheels[k].setProperties( wheelIds[(id*2)+(k-2)], rbclogic_type ); 
}

TTUEmulator::TTUEmulator( int id, const char * f_name, const char * rbclogic_type, 
                          const char * ttulogic_type, int mxw  ) :
  m_maxWheels{mxw},
  m_id{id},
  m_mode{1},
  m_line{1},
  m_ttuconf{std::make_unique<TTUBasicConfig>(ttulogic_type)},
  m_debug{false}
{
  
  for( int k=0; k < m_maxWheels; ++k ) 
    m_Wheels[k].setProperties( wheelIds[(id*2)+(k-2)], f_name, rbclogic_type );
}

//=============================================================================
void TTUEmulator::setSpecifications( const TTUBoardSpecs * ttuspecs, const RBCBoardSpecs * rbcspecs) 
{
  
  m_ttuconf   = std::make_unique<TTUBasicConfig>(ttuspecs);
  
  for( int k=0; k < m_maxWheels; ++k)
    m_Wheels[k].setSpecifications( rbcspecs );

  std::vector<TTUBoardSpecs::TTUBoardConfig>::const_iterator itr;
  itr = m_ttuconf->m_ttuboardspecs->m_boardspecs.begin();
  
  m_mode = (*itr).m_triggerMode;
  
}

bool TTUEmulator::initialise()
{
  bool status(false);
  for( int k=0; k < m_maxWheels; ++k)
    status = m_Wheels[k].initialise( );
  
  status = m_ttuconf->initialise( m_line , m_id );
  
  if ( !status ) { 
    if( m_debug ) std::cout << "TTUEmulator> Problem initialising the Configuration \n"; 
    return false; };
  
  return status;
  
}

void TTUEmulator::SetLineId( int line )
{
  m_line = line;
}

void TTUEmulator::emulate() 
{
  //... only for testing
  for( int k=0; k < m_maxWheels; ++k ) 
    m_Wheels[k].emulate();
  
}

void TTUEmulator::processTtu( RPCInputSignal * signal ) 
{
  
  //. 
  int bx(0);
  bool trg(false); 

  if( m_debug ) std::cout << "TTUEmulator::processTtu starts" << '\n';
  
  m_trigger.reset();
  m_triggerBx.clear();
  
  std::vector<int> bxVec;
  std::vector<int>::iterator bxItr;
  std::map<int,RBCInput*> * linkboardin;
  std::map<int,RBCInput*>::iterator inItr;
  
  linkboardin = dynamic_cast<RBCLinkBoardGLSignal*>( signal )->m_linkboardin;
  
  for( inItr = (*linkboardin).begin(); inItr != (*linkboardin).end(); ++inItr) 
  {
    
    if ( (*inItr).first < 0 ) bx = (int) ceil( (*inItr).first / 1000000.0 );
    else bx = (int) floor( (*inItr).first / 1000000.0 );
    bxVec.push_back(bx);
    
  }
  
  bxItr = unique (bxVec.begin(), bxVec.end());
  bxVec.resize(bxItr - bxVec.begin());
  
  m_triggerBxVec.reserve(m_triggerBxVec.size()+bxVec.size());
  for ( bxItr = bxVec.begin(); bxItr != bxVec.end(); ++bxItr) {
    
    for( int k=0; k < m_maxWheels; ++k )
    {
      
      if ( m_Wheels[k].process( (*bxItr) , (*linkboardin) ) ) {
        
        m_Wheels[k].createWheelMap();
        
        m_Wheels[k].retrieveWheelMap( (m_ttuin[k]) );
        
        //.. execute selected logic at Ttu level
        m_ttuconf->ttulogic()->run( (m_ttuin[k]) );
        
        //... and produce a Wheel level trigger
        trg = m_ttuconf->ttulogic()->isTriggered();
        
        m_trigger.set(k,trg);
        
        if( m_debug ) std::cout << "TTUEmulator::processTtu ttuid: " << m_id 
                                << " bx: "          << (*bxItr)
                                << " wheel: "       << m_Wheels[k].getid()
                                << " response: "    << trg << std::endl;
        
      }

      
    }

    auto& triggerResponse = m_triggerBxVec.emplace_back();
    
    triggerResponse.setTriggerBits( (*bxItr) , m_trigger );
    m_triggerBx[ (*bxItr) ] = m_trigger;
    
  }
  
  
  if( m_debug ) std::cout << "TTUEmulator::processTtu> size of trigger map " 
                          << m_triggerBx.size() << std::endl;
  
  
  if( m_debug ) std::cout << "TTUEmulator::processTtu> done with this TTU: " << m_id << std::endl;

  bxVec.clear();

  if( m_debug ) std::cout << "TTUEmulator::processTtu ends" << '\n';
    
}

void TTUEmulator::processTtu( RPCInputSignal * signal , int wedgeId ) 
{
  
  //. 
  int bx(0);
  bool trg(false); 
  
  if( m_debug ) std::cout << "TTUEmulator::processTtu( Pointing ) starts " << '\n';
  
  m_trigger.reset();
  m_triggerBx.clear();
  
  std::vector<int> bxVec;
  std::vector<int>::iterator bxItr;
  std::map<int,RBCInput*> * linkboardin;
  std::map<int,RBCInput*>::iterator inItr;
  
  linkboardin = dynamic_cast<RBCLinkBoardGLSignal*>( signal )->m_linkboardin;
  
  for( inItr = (*linkboardin).begin(); inItr != (*linkboardin).end(); ++inItr) 
  {
    
    if ( (*inItr).first < 0 ) bx = (int) ceil( (*inItr).first / 1000000.0 );
    else bx = (int) floor( (*inItr).first / 1000000.0 );
    bxVec.push_back(bx);
    
  }
  
  bxItr = unique (bxVec.begin(), bxVec.end());
  bxVec.resize(bxItr - bxVec.begin());
  
  m_triggerBxVec.reserve(m_triggerBxVec.size()+bxVec.size());

  for ( bxItr = bxVec.begin(); bxItr != bxVec.end(); ++bxItr) {
        
    for( int k=0; k < m_maxWheels; ++k )
    {
      
      if ( m_Wheels[k].process( (*bxItr) , (*linkboardin) ) ) { // <- this process uses the default RBC emulation but need a different logic
        
        m_Wheels[k].createWheelMap();
        
        m_Wheels[k].retrieveWheelMap( (m_ttuin[k]) );
        
        //.. execute selected logic at Ttu level
        m_ttuconf->ttulogic()->run( (m_ttuin[k]) , wedgeId );
        
        //... and produce a Wheel-Wedge level trigger
        trg = m_ttuconf->ttulogic()->isTriggered();
        
        m_trigger.set(k,trg);
        
        if( m_debug ) std::cout << "TTUEmulator::processTtu( Pointing ) ttuid: " << m_id 
                                << " bx: "          << (*bxItr)
                                << " wheel: "       << m_Wheels[k].getid()
                                << " response: "    << trg << std::endl;
        
      }
      
      
    }
    
    auto& triggerResponse = m_triggerBxVec.emplace_back();
    triggerResponse.setTriggerBits( (*bxItr) , wedgeId, m_trigger );
    m_triggerBx[ (*bxItr) ] = m_trigger;
  }
  
  if( m_debug ) std::cout << "TTUEmulator::processTtu (Pointing) > size of trigger map " 
                          << m_triggerBx.size() << std::endl;
  
  if( m_debug ) std::cout << "TTUEmulator::processTtu (Pointing) > done with this TTU: " << m_id << std::endl;
  
  bxVec.clear();

  if( m_debug ) std::cout << "TTUEmulator::processTtu( Pointing ) end" << '\n';
  
}


void TTUEmulator::clearTriggerResponse()
{
   m_triggerBxVec.clear();
}

//.................................................................

void TTUEmulator::printinfo() const 
{
  
  std::cout << "TTUEmulator: " << m_id << '\n';
  for( int k=0; k < m_maxWheels; ++k ) 
    m_Wheels[k].printinfo();
  
}

