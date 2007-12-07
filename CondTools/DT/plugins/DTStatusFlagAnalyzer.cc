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
#include "CondTools/DT/plugins/DTStatusFlagAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTStatusFlagHandler.h"
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
DTStatusFlagAnalyzer::DTStatusFlagAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTStatusFlag>( ps, "DTStatusFlag" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlagAnalyzer::~DTStatusFlagAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTStatusFlagAnalyzer::initSource( const edm::Event& evt,
                                       const edm::EventSetup& est ) {
  m_handler_object = new DTStatusFlagHandler( "DTStatusFlag",
                                              m_offline_connection,
                                              evt, est,
                                              dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTStatusFlagAnalyzer);

