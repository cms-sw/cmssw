// Include files 


// local
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLinkBoardGLSignal.h"

//-----------------------------------------------------------------------------
// Implementation file for class : ProcessTestSignal
//
// 2008-11-17 : Andres Osorio
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
ProcessTestSignal::ProcessTestSignal( const char * f_name ) 
  :m_in{}
{
  
  m_in.open(f_name);
  
  if(!m_in.is_open()) {
    std::cout << "ProcessTestSignal> cannot open file" << std::endl;
  } else { 
    std::cout << "ProcessTestSignal> file is now open" << std::endl;
  }
  
  m_lbin = std::make_unique<RBCLinkBoardGLSignal>( &m_data );
  
}
//=============================================================================
// Destructor
//=============================================================================
ProcessTestSignal::~ProcessTestSignal() {

  m_in.close();
} 

//=============================================================================
int ProcessTestSignal::next()
{
  
  reset();
  
  if ( m_in.fail() ) return 0;
  
  for(int j=0; j < 5; ++j) {
    
    auto& block = m_vecdata.emplace_back();
    (m_in) >> (*block);
  }
  
  builddata();
  
  if ( m_in.eof() ) return 0;
  return 1;
  
}

void ProcessTestSignal::showfirst() 
{
  rewind();
  for(auto& d: m_vecdata)
    std::cout << (*d);
  rewind();
  
}

void ProcessTestSignal::rewind() 
{ 
  m_in.clear();
  m_in.seekg(0,std::ios::beg); 
}

void ProcessTestSignal::reset()
{
  
   m_vecdata.clear();
  
}

void ProcessTestSignal::builddata() 
{
  
  int _code(0);
  for(auto& d : m_vecdata)
  {
    for(int k=0; k < 6; ++k) {
      
      _code = 10000*(d->m_wheel)
        + 100*d->m_sec1[k]
        + 1*d->m_sec2[k];
      RBCInput * _signal = & (d->m_orsignals[k]);
      _signal->needmapping = true;
      m_data.insert( std::make_pair( _code , _signal) );
      
    }
  }
  
}

