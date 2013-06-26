#include "CondCore/DBCommon/interface/DbConnection.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include <string>
#include <iostream>

void enableStatus( bool flag ){
  if( flag ){
    std::cout << " ENABLED;";
  } else {
    std::cout << " DISABLED;";
  }
}

void dumpConnectionConfiguration( const cond::DbConnectionConfiguration& conf ){
  std::cout << " Connection Sharing: ";
  enableStatus( conf.isConnectionSharingEnabled() );
  std::cout << std::endl;
  std::cout << " Connection TimeOut: "<<conf.connectionTimeOut()<< std::endl;
  std::cout << " ReadOnly Sessions on Update Connections: ";
  enableStatus( conf.isReadOnlySessionOnUpdateConnectionEnabled() );
  std::cout << std::endl;
  std::cout << " Connection RetrialPeriod: "<<conf.connectionRetrialPeriod()<< std::endl;
  std::cout << " Connection RetrialTimeOut: "<<conf.connectionRetrialTimeOut()<< std::endl;
  std::cout << " Pool Automatic CleanUp: ";
  enableStatus( conf.isPoolAutomaticCleanUpEnabled() );
  std::cout << std::endl;
  std::cout << " Coral Authentication Path:";
  if( conf.authenticationPath().empty()) std::cout << "NONE";
  else std::cout << conf.authenticationPath()<< std::endl;
  std::cout << std::endl;
  std::cout << " Coral Message Level: "<<conf.messageLevel()<< std::endl;
  std::cout << " SQL Monitoring: ";
  enableStatus( conf.isSQLMonitoringEnabled() );
  std::cout << std::endl;
}

int main(){
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection conn0;
  conn0.configure( cond::CmsDefaults );
  std::cout << "## configuration CmsDefaults"<<std::endl;
  dumpConnectionConfiguration( conn0.configuration() );
  conn0.close();
  cond::DbConnection conn1;
  conn1.configure( cond::CoralDefaults );
  std::cout << "## configuration CoralDefaults"<<std::endl;
  dumpConnectionConfiguration( conn1.configuration() );
  conn1.close();
  cond::DbConnection conn2;
  conn2.configure(  );
  std::cout << "## configuration basic"<<std::endl;
  dumpConnectionConfiguration( conn2.configuration() );
  conn2.close();
  cond::DbConnection conn;
  std::cout << "## checking connection open/close"<<std::endl;
  if(!conn.isOpen()){
    std::cout << "ERROR: connection is closed."<<std::endl;
  }
  std::cout << "## ok, connection is open"<<std::endl;
  conn = conn2;
  if(conn.isOpen()){
    std::cout << "ERROR: connection is still open."<<std::endl;
  }
  std::cout << "## ok, connection is closed"<<std::endl;
}
