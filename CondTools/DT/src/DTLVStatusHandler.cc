/*
 *  See header file for a description of this class.
 *
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


#include "CondCore/CondDB/interface/ConnectionPool.h"

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

#include <iostream>

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
 bufferConnect(         ps.getParameter<std::string> ( "bufferDB" ) ),
 omds_session(),
 buff_session() {
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
  std::cout << "configure omds DbConnection" << std::endl;
  cond::persistency::ConnectionPool connection;
  connection.setAuthenticationPath( onlineAuthentication );
  connection.configure();
  std::cout << "create omds DbSession" << std::endl;
  omds_session = connection.createSession( onlineConnect );
  std::cout << "start omds transaction" << std::endl;
  omds_session.transaction().start();
  std::cout << "" << std::endl;

  // buffer DB connection
  std::cout << "create buffer DbSession" << std::endl;
  cond::persistency::Session buff_session = connection.createSession(bufferConnect);
  std::cout << "start buffer transaction" << std::endl;
  buff_session.transaction().start();

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


