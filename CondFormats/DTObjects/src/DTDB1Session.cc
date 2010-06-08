/*
 *  See header file for a description of this class.
 *
 *  TEMPORARY TOOL TO HANDLE CONFIGURATIONS
 *  TO BE REMOVED IN FUTURE RELEASES
 *
 *
 *  $Date: 2009/06/05 09:31:22 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTDB1Session.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
//#include "CondCore/DBCommon/interface/PoolStorageManager.h"
//#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
//#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
//#include "CondCore/DBCommon/src/ServiceLoader.h"
//#include "RelationalAccess/IConnectionService.h"
//#include "RelationalAccess/IWebCacheControl.h"
#include <vector>

#include "CondCore/DBCommon/interface/SessionConfiguration.h"



//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTDB1Session::DTDB1Session( const std::string& dbFile,
                          const std::string& dbCatalog,
                          const std::string& auth_path,
                          bool siteLocalConfig ):
  m_session( 0 ),
  m_pooldb( 0 ) {

   std::string connect( dbFile );
   std::string catconnect( dbCatalog );
//   int connectionTimeOut = 600;
//   int connectionRetrialPeriod = 30;
//   int connectionRetrialTimeOut = 180;
   m_session = new cond::DBSession();
   if ( auth_path.empty() ) {
     m_session->configuration().setAuthenticationMethod( cond::Env );
   }
   else {
     m_session->configuration().setAuthenticationMethod( cond::XML );
     m_session->configuration().setAuthenticationPath( auth_path );
   }
   m_session->configuration().setMessageLevel( cond::Error );
/*
   m_session->configuration().connectionConfiguration()->enableConnectionSharing();
   m_session->configuration().connectionConfiguration()->setConnectionTimeOut(
                                           connectionTimeOut );
   m_session->configuration().connectionConfiguration()->
              enableReadOnlySessionOnUpdateConnections();
   m_session->configuration().connectionConfiguration()->
              setConnectionRetrialPeriod( connectionRetrialPeriod );
   m_session->configuration().connectionConfiguration()->
              setConnectionRetrialTimeOut( connectionRetrialTimeOut );
*/
   m_session->open();
   if ( siteLocalConfig ) {
    edm::Service<edm::SiteLocalConfig> localconfservice;
    if( !localconfservice.isAvailable() ){
      throw cms::Exception("edm::SiteLocalConfigService is not available");
    }
   }
//   m_connection = new cond::Connection( connect, -1 );
   if ( cond::ConnectionHandler::Instance().getConnection(connect) == 0 ) {
     std::cout << "register " << connect << std::endl;
     cond::ConnectionHandler::Instance().registerConnection(connect,*m_session,0);
   }
   else {
     std::cout << "already registered " << connect << std::endl;
   }
   m_connection = cond::ConnectionHandler::Instance().getConnection(connect);
//   m_pooldb = new cond::PoolStorageManager( connect, catconnect, m_session );

}

//--------------
// Destructor --
//--------------
DTDB1Session::~DTDB1Session() {
//  delete m_connection;
  delete m_session;
}

//--------------
// Operations --
//--------------
/// get storage manager
cond::PoolTransaction* DTDB1Session::poolDB() const {
//cond::PoolStorageManager* DTDB1Session::poolDB() const {
  return m_pooldb;
}


/// start transaction
void DTDB1Session::connect( bool readOnly ) {
  m_connection->connect( m_session );
  m_pooldb = &( m_connection->poolTransaction() );
  m_pooldb->start( readOnly );
//  m_pooldb->connect();
//  m_pooldb->startTransaction( readOnly );
  return;
}


/// end   transaction
void DTDB1Session::disconnect() {
  m_pooldb->commit();
  m_connection->disconnect();
//  m_pooldb->commit();
//  m_pooldb->disconnect();	
  return;
}
