#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include "testCondObj.h"
#include <string>
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection connection;
  connection.configuration().setMessageLevel( coral::Error );
  connection.configure();
  //
  cond::DbSession session = connection.createSession();
  session.open( "sqlite_file:testMultipleConnection0.db" );
  session.transaction().start(false);
  session.transaction().commit();
  boost::shared_ptr<testCondObj> myobj( new testCondObj );
  myobj->data.insert(std::make_pair<unsigned int,std::string>(10,"ten"));
  myobj->data.insert(std::make_pair<unsigned int,std::string>(2,"two"));
  session.transaction().start(false);
  session.createDatabase();
  session.storeObject(myobj.get(), "testCondObjContainer");
  session.transaction().commit();
  cond::DbSession session2 = connection.createSession();
  session2.open( "sqlite_file:testMultipleConnection1.db" );
  session2.transaction().start(false);
  coral::ISchema& schema = session2.nominalSchema();
  schema.dropIfExistsTable( "mytest" );
  coral::TableDescription description0;
  description0.setName( "mytest" );
  description0.insertColumn( "ID",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
  description0.insertColumn( "X",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
  description0.insertColumn( "Y",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
  description0.insertColumn( "ORDER",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
  std::vector<std::string> idx_cols;
  idx_cols.push_back("ORDER");
  description0.createIndex("IDX1",idx_cols,false);
  schema.createTable( description0 );
  std::set<std::string> result=schema.listTables();
  for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
    std::cout<<"table names: "<<*it<<std::endl;
  }    
  session2.transaction().commit();
}
