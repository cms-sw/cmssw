/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:04 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTPerformancePopConAnalyzer.h"

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
DTPerformancePopConAnalyzer::DTPerformancePopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTPerformance>( ps, "DTPerformance" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTPerformancePopConAnalyzer::~DTPerformancePopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTPerformancePopConAnalyzer::initSource( const edm::Event& evt,
                                              const edm::EventSetup& est ) {
  m_handler_object = new DTPerformanceHandler( "DTPerformance",
                                               m_offline_connection,
                                               evt, est,
                                               dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTPerformancePopConAnalyzer);

