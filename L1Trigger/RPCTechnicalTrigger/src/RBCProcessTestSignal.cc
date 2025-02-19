// $Id: RBCProcessTestSignal.cc,v 1.2 2009/05/16 19:43:32 aosorio Exp $
// Include files 


// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardSignal.h"
//-----------------------------------------------------------------------------
// Implementation file for class : RBCProcessTestSignal
//
// 2008-10-10 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RBCProcessTestSignal::RBCProcessTestSignal( const char * f_name )
{
  
  m_in = new std::ifstream();
  m_in->open(f_name);
  
  if(!m_in->is_open()) {
    std::cout << "RBCProcessTestSignal> cannot open file" << std::endl;
  } else { 
    std::cout << "RBCProcessTestSignal> file is now open" << std::endl;
  }
  
  m_input = new RBCInput();
  m_lbin  = dynamic_cast<RPCInputSignal*>( new RBCLinkBoardSignal( m_input ) );
  
  showfirst();

}
//=============================================================================
// Destructor
//=============================================================================
RBCProcessTestSignal::~RBCProcessTestSignal() 
{
  m_in->close();
  delete m_in;

  if ( m_lbin  ) delete m_lbin;
  
  if ( m_input ) delete m_input;
  
} 

//=============================================================================

int RBCProcessTestSignal::next()
{
  
  if ( m_in->fail()) return 0;
  (*m_in) >> (*m_input);
  if ( m_in->eof() ) return 0;
  return 1;
  
}

void RBCProcessTestSignal::showfirst() 
{
  rewind();
  (*m_in) >> (*m_input);
  std::cout << (*m_input);
  rewind();
  
}

void RBCProcessTestSignal::rewind() 
{ 
  m_in->clear();
  m_in->seekg(0,std::ios::beg); 
}

