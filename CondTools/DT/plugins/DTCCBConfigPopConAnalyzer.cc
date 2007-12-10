/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:12:45 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTCCBConfigPopConAnalyzer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTCCBConfigHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTCCBConfigPopConAnalyzer::DTCCBConfigPopConAnalyzer( const edm::ParameterSet& ps ):
 popcon::PopConAnalyzer<DTCCBConfig>( ps, "DTCCBConfig" ),
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 offlineAuthentication( ps.getParameter<edm::ParameterSet>( "DBParameters" )
                          .getUntrackedParameter<std::string> (
                        "authenticationPath" ) ),
// catalog(               ps.getParameter<std::string> ( "catalog" ) ),
 listToken(             ps.getParameter<std::string> ( "token" ) ) {
//  edm::ParameterSet pdb( ps.getParameter<edm::ParameterSet>(
//                         "DBParameters" ) );
//  offlineAuthentication = pdb.getUntrackedParameter<std::string> (
//                          "authenticationPath" );
//  std::cout << pdb.getUntrackedParameter<std::string> (
//               "authenticationPath" )
//            << std::endl;
  std::cout <<  onlineAuthentication << " "
            << offlineAuthentication << std::endl;
}

//--------------
// Destructor --
//--------------
DTCCBConfigPopConAnalyzer::~DTCCBConfigPopConAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTCCBConfigPopConAnalyzer::initSource( const edm::Event& evt,
                                            const edm::EventSetup& est ) {
  m_handler_object = new DTCCBConfigHandler( "DTCCBConfig",
                                             m_offline_connection,
                                             evt, est,
                                             dataTag,
                                             onlineConnect,
                                             onlineAuthentication,
                                             offlineAuthentication,
                                             listToken );
  return;
}

DEFINE_FWK_MODULE(DTCCBConfigPopConAnalyzer);

