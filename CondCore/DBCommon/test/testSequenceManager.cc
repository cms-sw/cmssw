#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SequenceManager.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include <string>
#include <iostream>
//#include <stdio.h>
//#include <time.h>

int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection connection;
  connection.configuration().setMessageLevel( coral::Error );
  connection.configure();
  cond::DbSession session = connection.createSession();
  session.open( "sqlite_file:testSequenceManager.db" );
  session.transaction().start(false);
  cond::SequenceManager sequenceGenerator(session,"mysequenceDepot");
  if( !sequenceGenerator.existSequencesTable() ){
    sequenceGenerator.createSequencesTable();
  }
  unsigned long long targetId=sequenceGenerator.incrementId("MYLOGDATA");
  std::cout<<"targetId for table MYLOGDATA "<<targetId<<std::endl;
  sequenceGenerator.clear();
  session.transaction().commit();
}
