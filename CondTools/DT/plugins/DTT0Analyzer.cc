/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/24 12:29:54 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTT0Analyzer.h"

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
DTT0Analyzer::DTT0Analyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTT0>( ps, "DTT0" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTT0Analyzer::~DTT0Analyzer() {
}

//--------------
// Operations --
//--------------
void DTT0Analyzer::initSource( const edm::Event& evt,
                               const edm::EventSetup& est ) {
  m_handler_object = new DTT0Handler( "DTT0",
                                      m_offline_connection,
                                      evt, est,
                                      dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTT0Analyzer);

