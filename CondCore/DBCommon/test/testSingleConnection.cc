#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "testCondObj.h"
#include <string>
#include <iostream>


#include <stdio.h>
#include <time.h>
void wait ( int seconds )
{
  clock_t endwait;
  endwait=clock()+seconds*CLOCKS_PER_SEC;
  while (clock() < endwait) {}
}
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection connection;
  connection.configuration().setMessageLevel( coral::Error );
  connection.configuration().setPoolAutomaticCleanUp( false );
  connection.configuration().setConnectionTimeOut(0);
  connection.configure();
  cond::DbSession session = connection.createSession();
  session.open( "sqlite_file:testSingleConnection.db" );
  boost::shared_ptr<testCondObj> myobj( new testCondObj );
  myobj->data.insert(std::make_pair<unsigned int,std::string>(10,"ten"));
  myobj->data.insert(std::make_pair<unsigned int,std::string>(2,"two"));
  session.transaction().start(false);
  session.createDatabase();
  std::cout<<"waiting for 5 sec in pool transaction..."<<std::endl;
  wait(5);
  session.storeObject( myobj.get(), "testCondObjContainer" );
  session.transaction().commit();
  std::cout<<"waiting for 5 sec in coral transaction..."<<std::endl;
  wait(5);
  session.transaction().start(false);
  std::set<std::string> result=session.nominalSchema().listTables();
  session.transaction().commit();
  for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
    std::cout<<"table name "<<*it<<std::endl;
  }
}
