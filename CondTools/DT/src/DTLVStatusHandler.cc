/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/25 11:00:00 $
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
#include "CondTools/DT/interface/DTDBSession.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
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

  cond::DBSession* omds_session;
  cond::Connection* omds_connect;
  cond::CoralTransaction* omds_transaction;

  omds_session = new cond::DBSession();
  // to get the username/passwd from $CORAL_AUTH_PATH/authentication.xml
  omds_session->configuration().setAuthenticationMethod( cond::XML );
  omds_session->configuration().setAuthenticationPath( onlineAuthentication );
  // set message level to Error or Debug
  omds_session->configuration().setMessageLevel( cond::Error );
//  omds_session->connectionConfiguration().setConnectionRetrialTimeOut( 60 );
  omds_session->open();


  omds_connect = new cond::Connection( onlineConnect );
  omds_connect->connect( omds_session );
  omds_transaction = &( omds_connect->coralTransaction() );
  omds_transaction->start( true );

//  std::cout << "get session proxy... " << std::endl;
  omds_s_proxy = &( omds_transaction->coralSessionProxy() );
//  std::cout << "session proxy got" << std::endl;
  omds_transaction->start( true );

// buffer DB connection

  cond::DBSession* buff_session;
  cond::Connection* buff_connect;
  cond::CoralTransaction* buff_transaction;

  buff_session = new cond::DBSession();
  // to get the username/passwd from $CORAL_AUTH_PATH/authentication.xml
  buff_session->configuration().setAuthenticationMethod( cond::XML );
  buff_session->configuration().setAuthenticationPath( onlineAuthentication );
  // set message level to Error or Debug
  buff_session->configuration().setMessageLevel( cond::Error );
//  buff_session->connectionConfiguration().setConnectionRetrialTimeOut( 60 );
  buff_session->open();

  buff_connect = new cond::Connection( bufferConnect );
  buff_connect->connect( buff_session );
  buff_transaction = &( buff_connect->coralTransaction() );
  buff_transaction->start( false );

//  std::cout << "get session proxy... " << std::endl;
  buff_s_proxy = &( buff_transaction->coralSessionProxy() );
//  std::cout << "session proxy got" << std::endl;
  buff_transaction->start( false );

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

  delete omds_connect;
  delete omds_session;
  buff_transaction->commit();
  delete buff_connect;
  delete buff_session;

  return;

}


std::string DTLVStatusHandler::id() const {
  return dataTag;
}


