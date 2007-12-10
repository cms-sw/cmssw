/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:10 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTReadOutMappingPopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTReadOutMappingHandler.h"
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
DTReadOutMappingPopConAnalyzer::DTReadOutMappingPopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTReadOutMapping>( ps, "DTReadOutMapping" ),
 dataTag(  ps.getParameter<std::string> ( "tag" ) ),
 fileName( ps.getParameter<std::string> ( "file" ) ) {
}

//--------------
// Destructor --
//--------------
DTReadOutMappingPopConAnalyzer::~DTReadOutMappingPopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTReadOutMappingPopConAnalyzer::initSource( const edm::Event& evt,
                                                 const edm::EventSetup& est ) {
  m_handler_object = new DTReadOutMappingHandler( "DTReadOutMapping",
                                                  m_offline_connection,
                                                  evt, est,
                                                  dataTag, fileName );
  return;
}

DEFINE_FWK_MODULE(DTReadOutMappingPopConAnalyzer);

