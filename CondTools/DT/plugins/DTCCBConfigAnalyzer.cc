/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/24 12:29:52 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTCCBConfigAnalyzer.h"

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
DTCCBConfigAnalyzer::DTCCBConfigAnalyzer( const edm::ParameterSet& ps ):
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
DTCCBConfigAnalyzer::~DTCCBConfigAnalyzer() {
}

//--------------
// Operations --
//--------------
void DTCCBConfigAnalyzer::initSource( const edm::Event& evt,
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

DEFINE_FWK_MODULE(DTCCBConfigAnalyzer);

