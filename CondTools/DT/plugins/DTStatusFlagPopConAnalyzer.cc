/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:13 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTStatusFlagPopConAnalyzer.h"

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
DTStatusFlagPopConAnalyzer::DTStatusFlagPopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTStatusFlag>( ps, "DTStatusFlag" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlagPopConAnalyzer::~DTStatusFlagPopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTStatusFlagPopConAnalyzer::initSource( const edm::Event& evt,
                                             const edm::EventSetup& est ) {
  m_handler_object = new DTStatusFlagHandler( "DTStatusFlag",
                                              m_offline_connection,
                                              evt, est,
                                              dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTStatusFlagPopConAnalyzer);

