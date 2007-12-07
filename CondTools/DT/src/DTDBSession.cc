/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/24 12:29:54 $
 *  $Revision: 1.1.4.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTDBSession.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
//#include "CondCore/DBCommon/interface/PoolStorageManager.h"
//#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
//#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
//#include "CondCore/DBCommon/src/ServiceLoader.h"
#include "CondCore/DBOutputService/interface/Exception.h"
#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/DBOutputService/src/serviceCallbackToken.h"
//#include "RelationalAccess/IConnectionService.h"
//#include "RelationalAccess/IWebCacheControl.h"
#include <vector>

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"



//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTDBSession::DTDBSession( const std::string& dbFile,
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
   m_connection = new cond::Connection( connect, -1 );
//   m_pooldb = new cond::PoolStorageManager( connect, catconnect, m_session );

}

//--------------
// Destructor --
//--------------
DTDBSession::~DTDBSession() {
  delete m_connection;
  delete m_session;
}

//--------------
// Operations --
//--------------
/// get storage manager
cond::PoolTransaction* DTDBSession::poolDB() const {
//cond::PoolStorageManager* DTDBSession::poolDB() const {
  return m_pooldb;
}


/// start transaction
void DTDBSession::connect( bool readOnly ) {
  m_connection->connect( m_session );
  m_pooldb = &( m_connection->poolTransaction() );
  m_pooldb->start( readOnly );
//  m_pooldb->connect();
//  m_pooldb->startTransaction( readOnly );
  return;
}


/// end   transaction
void DTDBSession::disconnect() {
  m_pooldb->commit();
  m_connection->disconnect();
//  m_pooldb->commit();
//  m_pooldb->disconnect();	
  return;
}
