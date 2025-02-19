#include "CondTools/L1Trigger/interface/DataManager.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"





namespace l1t
{

  DataManager::DataManager()
    : session( 0 ),
      connection( 0 ) {}

  DataManager::DataManager (const std::string & connectString,
			    const std::string & authenticationPath,
			    bool isOMDS )
  {
    connect( connectString, authenticationPath, isOMDS ) ;
  }

  void
  DataManager::connect(const std::string & connectString,
		       const std::string & authenticationPath,
		       bool isOMDS )
  {
    connection = new cond::DbConnection() ;
    setDebug( false ) ;

    if( !isOMDS )
      {
	connection->configuration().setConnectionSharing( true ) ;
	connection->configuration().setReadOnlySessionOnUpdateConnections(
	  true ) ;
      }

    connection->configuration().setAuthenticationPath( authenticationPath ) ;
    connection->configure() ;

    session = new cond::DbSession( connection->createSession() ) ;
    session->open( connectString, isOMDS ) ;
}

DataManager::~DataManager ()
{
  // delete all in reverse direction
  if( connection )
    connection->close() ;
  if( session )
    session->close() ;

  delete connection ;
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
	connection->configuration().setMessageLevel( coral::Debug ) ;
      }
    else
      {
	connection->configuration().setMessageLevel( coral::Error ) ;
      }
  }
}
