// Include files 



// local
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUConfigurator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//-----------------------------------------------------------------------------
// Implementation file for class : TTUConfigurator
//
// 2009-06-02 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
TTUConfigurator::TTUConfigurator( const std::string& infile ):
  m_in{},
  m_rbcspecs{},
  m_ttuspecs{}
{
  
  m_in.open( infile.c_str() );
  
  if(!m_in.is_open()) {
    edm::LogError("TTUConfigurator") << "TTUConfigurator cannot open file";
    m_hasConfig = false;
  } else {
    m_hasConfig = true;
  }
}
//=============================================================================
// Destructor
//=============================================================================
TTUConfigurator::~TTUConfigurator() {
  m_in.close();  
} 

//=============================================================================

void TTUConfigurator::process()
{
  
  addData( m_rbcspecs );
  addData( m_ttuspecs );
  
}

void TTUConfigurator::addData( RBCBoardSpecs& specs )
{
  specs.v_boardspecs.reserve(30);
  for( int i=0; i < 30; i++) {
    auto& board = specs.v_boardspecs.emplace_back();
    m_in >> board;
  }
  
}

void TTUConfigurator::addData( TTUBoardSpecs& specs )
{
  specs.m_boardspecs.reserve(3);
  for(int i=0; i < 3; i++){
    auto& board = specs.m_boardspecs.emplace_back();
    
    m_in >> board;
  }
}
