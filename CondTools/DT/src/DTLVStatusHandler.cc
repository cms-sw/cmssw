/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/26 14:11:04 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTLVStatusHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTLVStatusHandler::DTLVStatusHandler( const edm::ParameterSet& ps ):
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 bufferConnect(         ps.getParameter<std::string> ( "bufferDB" ) ) {
  std::cout << " PopCon application for DT DCS data (CCB status) export "
            << std::endl;
}

//--------------
// Destructor --
//--------------
DTLVStatusHandler::~DTLVStatusHandler() {
}

//--------------
// Operations --
//--------------
void DTLVStatusHandler::getNewObjects() {

// online DB connection
  std::cout << "create omds DbConnection" << std::endl;
  cond::DbConnection* omds_conn = new cond::DbConnection;
  std::cout << "configure omds DbConnection" << std::endl;
//  conn->configure( cond::CmsDefaults );
  omds_conn->configuration().setAuthenticationPath( onlineAuthentication );
  omds_conn->configure();
  std::cout << "create omds DbSession" << std::endl;
  cond::DbSession omds_session = omds_conn->createSession();
  std::cout << "open omds session" << std::endl;
  omds_session.open( onlineConnect );
  std::cout << "start omds transaction" << std::endl;
  omds_session.transaction().start();
  std::cout << "create omds coralSession" << std::endl;
  omds_s_proxy = &( omds_session.coralSession() );
  std::cout << "" << std::endl;

// buffer DB connection
  std::cout << "create buffer DbConnection" << std::endl;
  cond::DbConnection* buff_conn = new cond::DbConnection;
  std::cout << "configure buffer DbConnection" << std::endl;
//  conn->configure( cond::CmsDefaults );
  buff_conn->configuration().setAuthenticationPath( onlineAuthentication );
  buff_conn->configure();
  std::cout << "create buffer DbSession" << std::endl;
  cond::DbSession buff_session = buff_conn->createSession();
  std::cout << "open buffer session" << std::endl;
  buff_session.open( bufferConnect );
  std::cout << "start buffer transaction" << std::endl;
  buff_session.transaction().start();
  std::cout << "create buffer coralSession" << std::endl;
  buff_s_proxy = &( buff_session.coralSession() );
  std::cout << "buffer session proxy got" << std::endl;

// offline info

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
  unsigned int last = ti.lastInterval.first;
  std::cout << "latest DCS data (CCB status) already copied for run: "
            << last << std::endl;

  if ( last == 0 ) {
    DTLVStatus* dummyConf = new DTLVStatus( dataTag );
    cond::Time_t snc = 1;
    m_to_transfer.push_back( std::make_pair( dummyConf, snc ) );
  }

  //to access the information on last successful log entry for this tag:
//  cond::LogDBEntry const & lde = logDBEntry();     

  //to access the lastest payload (Ref is a smart pointer)
//  Ref payload = lastPayload();

  unsigned lastRun = last;
  std::cout << "check for new runs since " << lastRun << std::endl;

  buff_session.transaction().commit();
  buff_session.close();
  omds_session.close();

  return;

}


std::string DTLVStatusHandler::id() const {
  return dataTag;
}


