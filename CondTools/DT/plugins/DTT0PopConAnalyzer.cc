/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:14 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTT0PopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTT0Handler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTT0PopConAnalyzer::DTT0PopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTT0>( ps, "DTT0" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTT0PopConAnalyzer::~DTT0PopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTT0PopConAnalyzer::initSource( const edm::Event& evt,
                                     const edm::EventSetup& est ) {
  m_handler_object = new DTT0Handler( "DTT0",
                                      m_offline_connection,
                                      evt, est,
                                      dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTT0PopConAnalyzer);

