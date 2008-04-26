#include "CondTools/L1Trigger/interface/DataManager.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"


namespace l1t
{

DataManager::DataManager (const std::string & connect,
			  const std::string & authenticationPath,
			  bool isOMDS )
{
//     // Initialize session used with this object
//     poolSession = new cond::DBSession ();
//     poolSession->configuration ().setAuthenticationMethod (cond::Env);
// //    poolSession->sessionConfiguration ().setMessageLevel (cond::Info);
//     poolSession->configuration ().connectionConfiguration ()->enableConnectionSharing ();
//     poolSession->configuration ().connectionConfiguration ()->enableReadOnlySessionOnUpdateConnections ();

//     coralSession = new cond::DBSession ();
//     coralSession->configuration ().setAuthenticationMethod (cond::Env);
// //    coralSession->sessionConfiguration ().setMessageLevel (cond::Info);
//     coralSession->configuration ().connectionConfiguration ()->enableConnectionSharing ();
//     coralSession->configuration ().connectionConfiguration ()->enableReadOnlySessionOnUpdateConnections ();

    session = new cond::DBSession();
    setDebug( false ) ;

    if( isOMDS )
      {
	session->configuration().setAuthenticationMethod(cond::XML);
      }
    else
      {
	//	session->configuration().setAuthenticationMethod(cond::Env);
	session->configuration().setAuthenticationMethod(cond::XML);
	session->configuration().connectionConfiguration()
	  ->enableConnectionSharing ();
	session->configuration().connectionConfiguration()
	  ->enableReadOnlySessionOnUpdateConnections ();
      }
    session->configuration().setAuthenticationPath( authenticationPath ) ;
    session->open() ;
    

    // create Coral connection and pool. This ones should be connected on required basis
//     coral = new cond::CoralTransaction (connect, coralSession);
//     pool = new cond::PoolTransaction (connect, catalog, poolSession);

    connection = new cond::Connection( connect ) ;
    connection->connect( session ) ;
    cond::CoralTransaction& coral = connection->coralTransaction() ;
//     coral = &( connection->coralTransaction() ) ;
//     pool = &( connection->poolTransaction() ) ;

    // and data object
    metadata = new cond::MetaData (coral);

    // wsun: need to disconnect?
    // connection->disconnect() ;

    // can I use two sessions? One for pool and one for coral? So that I could connect to one and another....
    // will sqlite will be happy with transactions?
}

DataManager::~DataManager ()
{
  connection->disconnect() ;

    // detele all in reverse direction
    // closing and comminting may be a good idea here
    // in case we have exception in some other part between connect/close pairs

    delete metadata;
//     delete pool;
//     delete coral;
    delete connection ;
//     delete coralSession;
//     delete poolSession;
    delete session ;
}

edm::eventsetup::TypeTag DataManager::findType (const std::string & type) const
{
  static edm::eventsetup::TypeTag defaultType;
  edm::eventsetup::TypeTag typeTag = edm::eventsetup::TypeTag::findType (type);
  
  if (typeTag == defaultType)
    //throw cond::Exception ("l1t::DataManager::findType")
    edm::LogError("L1TriggerDB") << "DataManager::findType() : " << type << " was not found";
  
  return typeTag;
}

  void
  DataManager::setDebug( bool debug )
  {
    if( debug )
      {
	session->configuration().setMessageLevel( cond::Debug ) ;
      }
    else
      {
	session->configuration().setMessageLevel( cond::Error ) ;
      }
  }
}
