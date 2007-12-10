/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:01 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTMtimePopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTMtimeHandler.h"
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
DTMtimePopConAnalyzer::DTMtimePopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTMtime>( ps, "DTMtime" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTMtimePopConAnalyzer::~DTMtimePopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTMtimePopConAnalyzer::initSource( const edm::Event& evt,
                                        const edm::EventSetup& est ) {
  m_handler_object = new DTMtimeHandler( "DTMtime",
                                         m_offline_connection,
                                         evt, est,
                                         dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTMtimePopConAnalyzer);

