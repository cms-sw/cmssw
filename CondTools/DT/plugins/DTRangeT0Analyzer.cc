/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/24 12:29:53 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTRangeT0Analyzer.h"

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
DTRangeT0Analyzer::DTRangeT0Analyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTRangeT0>( ps, "DTRangeT0" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTRangeT0Analyzer::~DTRangeT0Analyzer() {
}

//--------------
// Operations --
//--------------
void DTRangeT0Analyzer::initSource( const edm::Event& evt,
                                    const edm::EventSetup& est ) {
  m_handler_object = new DTRangeT0Handler( "DTRangeT0",
                                           m_offline_connection,
                                           evt, est,
                                           dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTRangeT0Analyzer);

