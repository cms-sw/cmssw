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
#include "CondTools/DT/plugins/DTPerformanceAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTPerformanceHandler.h"
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
DTPerformanceAnalyzer::DTPerformanceAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTPerformance>( ps, "DTPerformance" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTPerformanceAnalyzer::~DTPerformanceAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTPerformanceAnalyzer::initSource( const edm::Event& evt,
                                        const edm::EventSetup& est ) {
  m_handler_object = new DTPerformanceHandler( "DTPerformance",
                                               m_offline_connection,
                                               evt, est,
                                               dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTPerformanceAnalyzer);

