/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:12:55 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTConfigDBInit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTDBSession.h"
#include "CondFormats/DTObjects/interface/DTConfigList.h"
#include "CondFormats/DTObjects/interface/DTConfigData.h"
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
DTConfigDBInit::DTConfigDBInit( const edm::ParameterSet& ps ):
 name(     ps.getParameter<std::string> ( "name"     ) ),
 contact(  ps.getParameter<std::string> ( "contact"  ) ),
 catalog(  ps.getParameter<std::string> ( "catalog"  ) ),
 authPath( ps.getParameter<std::string> ( "authPath" ) ) {
}

//--------------
// Destructor --
//--------------
DTConfigDBInit::~DTConfigDBInit() {
}

//--------------
// Operations --
//--------------
void DTConfigDBInit::beginJob() {

  DTDBSession* session = new DTDBSession( contact, catalog, authPath );
  session->connect( false );

  int dummyId = -999999999;
  DTConfigData* dummyConf = new DTConfigData();
  dummyConf->setId( dummyId );
  cond::TypedRef<DTConfigData> confRef( *session->poolDB(), dummyConf );
  confRef.markWrite( "DTConfigData" );

  DTConfigList* confList = new DTConfigList( name );
  DTConfigToken token;
  token.id  = 0;
  token.ref = confRef.token();
  confList->set( dummyId, token );
  cond::TypedRef<DTConfigList> setRef( *session->poolDB(), confList );
  setRef.markWrite( "DTConfigList" );
  std::string listToken = setRef.token();
  std::cout << "configuration list stored with token: " << std::endl 
            << listToken << std::endl;

  session->disconnect();
  delete session;
  return;

}


void DTConfigDBInit::analyze( const edm::Event& e,
                              const edm::EventSetup& c ) {
  return;
}

DEFINE_FWK_MODULE(DTConfigDBInit);

