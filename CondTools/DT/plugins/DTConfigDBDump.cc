/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:12:51 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTConfigDBDump.h"

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
DTConfigDBDump::DTConfigDBDump( const edm::ParameterSet& ps ):
 contact(  ps.getParameter<std::string> ( "contact"  ) ),
 catalog(  ps.getParameter<std::string> ( "catalog"  ) ),
 token(    ps.getParameter<std::string> ( "token"    ) ),
 authPath( ps.getParameter<std::string> ( "authPath" ) ) {
}

//--------------
// Destructor --
//--------------
DTConfigDBDump::~DTConfigDBDump() {
}

//--------------
// Operations --
//--------------
void DTConfigDBDump::beginJob() {

  DTDBSession* session = new DTDBSession( contact, catalog, authPath );
  session->connect( false );

  const DTConfigList* confList = 0;
  DTConfigHandler* confHandler = DTConfigHandler::create( session, token );
  confList = confHandler->getContainer();
  if ( confList != 0 ) {
    std::cout << confList->version() << std::endl;
    DTConfigList::const_iterator iter = confList->begin();
    DTConfigList::const_iterator iend = confList->end();
    std::cout << std::distance( iter, iend )
              << " configurations in the list " << confList << std::endl;
    while ( iter != iend ) {
      int confId = iter->first;
      std::cout << confId << " -> "
                << iter->second.ref << std::endl;
      std::cout << "========> " << std::endl;
      DTConfigData* configPtr;
      confHandler->get( confId, configPtr );
      std::cout << confId << "--->" << std::endl;
      DTConfigData::data_iterator d_iter = configPtr->dataBegin();
      DTConfigData::data_iterator d_iend = configPtr->dataEnd();
      while ( d_iter != d_iend ) std::cout << "    " << *d_iter++
                                           << std:: endl;
      DTConfigData::link_iterator l_iter = configPtr->linkBegin();
      DTConfigData::link_iterator l_iend = configPtr->linkEnd();
      while ( l_iter != l_iend ) std::cout << "     + " << *l_iter++
                                           << std:: endl;
      std::vector<const std::string*> list;
      confHandler->getData( confId, list );
      std::vector<const std::string*>::const_iterator s_iter = list.begin();
      std::vector<const std::string*>::const_iterator s_iend = list.end();
      while ( s_iter != s_iend ) std::cout << "        ----> "
                                           << **s_iter++ << std::endl;
      iter++;
    }
  }

  DTConfigHandler::remove( session );
  session->disconnect();
  delete session;

  return;

}


void DTConfigDBDump::analyze( const edm::Event& e,
                              const edm::EventSetup& c ) {
  return;
}

DEFINE_FWK_MODULE(DTConfigDBDump);

