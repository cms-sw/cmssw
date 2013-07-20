// $Id: TTUConfigurator.cc,v 1.3 2009/12/25 06:24:34 elmer Exp $
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
TTUConfigurator::TTUConfigurator( const std::string& infile ) {
  
  m_in = new std::ifstream();
  m_in->open( infile.c_str() );
  
  if(!m_in->is_open()) {
    edm::LogError("TTUConfigurator") << "TTUConfigurator cannot open file";
    m_hasConfig = false;
  } else {
    m_hasConfig = true;
  }
  
  m_rbcspecs = new RBCBoardSpecs();
  m_ttuspecs = new TTUBoardSpecs();
  
}
//=============================================================================
// Destructor
//=============================================================================
TTUConfigurator::~TTUConfigurator() {
  
  if ( m_in ) {
    m_in->close();
    delete m_in;
  }
  
  if ( m_rbcspecs ) delete m_rbcspecs;
  if ( m_ttuspecs ) delete m_ttuspecs;
  
} 

//=============================================================================

void TTUConfigurator::process()
{
  
  addData( m_rbcspecs );
  addData( m_ttuspecs );
  
}

void TTUConfigurator::addData( RBCBoardSpecs * specs )
{
  
  RBCBoardSpecs::RBCBoardConfig * board;
  
  for( int i=0; i < 30; i++) {
    
    board = new RBCBoardSpecs::RBCBoardConfig();
    
    (*m_in) >> (*board);
    
    specs->v_boardspecs.push_back( *board );
    
  }
  
}

void TTUConfigurator::addData( TTUBoardSpecs * specs )
{
  
  TTUBoardSpecs::TTUBoardConfig * board;
  
  for(int i=0; i < 3; i++){
    
    board= new TTUBoardSpecs::TTUBoardConfig();
    
    (*m_in) >> (*board);
    
    specs->m_boardspecs.push_back( *board );
    
  }
  
}
