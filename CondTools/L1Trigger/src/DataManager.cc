#include "CondTools/L1Trigger/interface/DataManager.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"





namespace l1t
{

  DataManager::DataManager()
    : session( 0 ){}

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
    setDebug( false ) ;
    cond::persistency::ConnectionPool connection;
    connection.setAuthenticationPath( authenticationPath ) ;
    if( debugFlag ) connection.setMessageVerbosity( coral::Debug );
    else connection.setMessageVerbosity( coral::Error ) ;
    connection.configure() ;

    session = connection.createSession( connectString, isOMDS );
}

DataManager::~DataManager ()
{
  // delete all in reverse direction
    session.close() ;

}

edm::eventsetup::TypeTag DataManager::findType (const std::string & type) 
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
    debugFlag = debug;
  }
}
