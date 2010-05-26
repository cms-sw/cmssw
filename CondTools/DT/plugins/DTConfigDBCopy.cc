/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:12:49 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTConfigDBCopy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTConfigHandler.h"
#include "CondTools/DT/interface/DTDBSession.h"
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
DTConfigDBCopy::DTConfigDBCopy( const edm::ParameterSet& ps ):
 sourceContact(  ps.getParameter<std::string> ( "sourceContact"  ) ),
 sourceCatalog(  ps.getParameter<std::string> ( "sourceCatalog"  ) ),
 sourceToken(    ps.getParameter<std::string> ( "sourceToken"    ) ),
 sourceAuthPath( ps.getParameter<std::string> ( "sourceAuthPath" ) ),
 targetContact(  ps.getParameter<std::string> (  "targetContact" ) ),
 targetCatalog(  ps.getParameter<std::string> (  "targetCatalog" ) ),
 targetToken(    ps.getParameter<std::string> (    "targetToken" ) ),
 targetAuthPath( ps.getParameter<std::string> ( "targetAuthPath" ) ) {
}

//--------------
// Destructor --
//--------------
DTConfigDBCopy::~DTConfigDBCopy() {
}

//--------------
// Operations --
//--------------
void DTConfigDBCopy::beginJob() {

  DTDBSession* sourceSession = new DTDBSession( sourceContact,
                                                sourceCatalog,
                                                sourceAuthPath );
  DTDBSession* targetSession = new DTDBSession( targetContact,
                                                targetCatalog,
                                                targetAuthPath );
  sourceSession->connect( false );
  targetSession->connect( false );

  const DTConfigList* confList = 0;

  DTConfigHandler* confHandler =
  DTConfigHandler::create( sourceSession, sourceToken );
  confList = confHandler->getContainer();

  std::string cloneToken;
  if ( confHandler != 0 ) {
    if ( targetToken.length() ) {
      cloneToken = confHandler->clone( targetSession, targetToken,
                                       "DTConfigList", "DTConfigData" );
      if ( cloneToken.length() ) std::cout << "unvalid target token, new "
                                           << "configuration list created"
                                           << std::endl;
      else                       std::cout << "existing "
                                           << "configuration list updated"
                                           << std::endl;
    }
    else {
      cloneToken = confHandler->clone( targetSession, "",
                                       "DTConfigList", "DTConfigData" );
    }
  }
  if ( cloneToken.length() ) 
       std::cout << "configuration list copied with token: " << cloneToken
                 << std::endl;
  else
       std::cout << "configuration list updated"
                 << std::endl;

  DTConfigHandler::remove( sourceSession );
  sourceSession->disconnect();
  targetSession->disconnect();
  delete sourceSession;
  delete targetSession;

  return;

}


void DTConfigDBCopy::analyze( const edm::Event& e,
                              const edm::EventSetup& c ) {
  return;
}

DEFINE_FWK_MODULE(DTConfigDBCopy);

