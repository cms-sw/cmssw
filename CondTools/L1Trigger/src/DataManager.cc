#include "CondTools/L1Trigger/interface/DataManager.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"


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
    // session->configuration().setBlobStreamer("COND/Services/TBufferBlobStreamingService") ;
    session->open() ;
    
    connection = new cond::Connection( connectString ) ;
    connection->connect( session ) ;
}

DataManager::~DataManager ()
{
  // delete all in reverse direction
  if( connection )
    connection->disconnect() ;

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
	session->configuration().setMessageLevel( cond::Debug ) ;
      }
    else
      {
	session->configuration().setMessageLevel( cond::Error ) ;
      }
  }
}
