/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:15 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTTtrigPopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTTtrigHandler.h"
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
DTTtrigPopConAnalyzer::DTTtrigPopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTTtrig>( ps, "DTTtrig" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTTtrigPopConAnalyzer::~DTTtrigPopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTTtrigPopConAnalyzer::initSource( const edm::Event& evt,
                                        const edm::EventSetup& est ) {
  m_handler_object = new DTTtrigHandler( "DTTtrig",
                                         m_offline_connection,
                                         evt, est,
                                         dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTTtrigPopConAnalyzer);

