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
#include "CondTools/DT/plugins/DTDeadFlagAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTDeadFlagHandler.h"
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
DTDeadFlagAnalyzer::DTDeadFlagAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTDeadFlag>( ps, "DTDeadFlag" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTDeadFlagAnalyzer::~DTDeadFlagAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTDeadFlagAnalyzer::initSource( const edm::Event& evt,
                                  const edm::EventSetup& est ) {
  m_handler_object = new DTDeadFlagHandler( "DTDeadFlag",
                                            m_offline_connection,
                                            evt, est,
                                            dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTDeadFlagAnalyzer);


