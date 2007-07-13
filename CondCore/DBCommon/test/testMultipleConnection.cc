#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/TypedRef.h"

#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/IPrimaryKey.h"
#include "RelationalAccess/IForeignKey.h"
#include "RelationalAccess/IIndex.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include "testCondObj.h"
#include <string>
#include <iostream>
int main(){
  cond::DBSession* session=new cond::DBSession;
  session->configuration().setMessageLevel( cond::Error );
  session->configuration().setAuthenticationMethod(cond::XML);
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  conHandler.registerConnection("mysqlite1","sqlite_file:mydata.db","file:mycatalog.xml",0);
  conHandler.registerConnection("mysqlite2","sqlite_file:miodati.db","file:mycatalog.xml",0);
  session->open();
  conHandler.connect(session);
  cond::Connection* myconnection=conHandler.getConnection("mysqlite1");
  std::cout<<"myconnection "<<myconnection<<std::endl;
  cond::CoralTransaction& coralTransaction=myconnection->coralTransaction(false);
  coralTransaction.start();
  coralTransaction.commit();
  testCondObj* myobj=new testCondObj;
  myobj->data.insert(std::make_pair<unsigned int,std::string>(10,"ten"));
  myobj->data.insert(std::make_pair<unsigned int,std::string>(2,"two"));
  cond::PoolTransaction& poolTransaction=myconnection->poolTransaction(false);
  poolTransaction.start();
  cond::TypedRef<testCondObj> myref(myconnection->poolTransaction(false),myobj);
  myref.markWrite("testCondObjContainer");
  poolTransaction.commit();
  cond::Connection* myconnection2=conHandler.getConnection("mysqlite2",false);
  std::cout<<"myconnection2 "<<myconnection2<<std::endl;
  cond::CoralTransaction& coralTransaction2=myconnection2->coralTransaction(false);
  coralTransaction2.start();
  coral::ISchema& schema = coralTransaction2.nominalSchema();
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
  coralTransaction2.commit();
  conHandler.disconnectAll();
  delete session;
}
