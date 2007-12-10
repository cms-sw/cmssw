/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:07 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTRangeT0PopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTRangeT0Handler.h"
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
DTRangeT0PopConAnalyzer::DTRangeT0PopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTRangeT0>( ps, "DTRangeT0" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTRangeT0PopConAnalyzer::~DTRangeT0PopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTRangeT0PopConAnalyzer::initSource( const edm::Event& evt,
                                          const edm::EventSetup& est ) {
  m_handler_object = new DTRangeT0Handler( "DTRangeT0",
                                           m_offline_connection,
                                           evt, est,
                                           dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTRangeT0PopConAnalyzer);

