#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"

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
  cond::Connection con("sqlite_file:mydata.db",0);
  session->open();
  con.connect(session);
  testCondObj* myobj=new testCondObj;
  myobj->data.insert(std::make_pair<unsigned int,std::string>(10,"ten"));
  myobj->data.insert(std::make_pair<unsigned int,std::string>(2,"two"));
  cond::PoolTransaction& poolTransaction=con.poolTransaction();
  poolTransaction.start(false);
  cond::TypedRef<testCondObj> myref(poolTransaction,myobj);
  myref.markWrite("testCondObjContainer");
  poolTransaction.commit();
  cond::CoralTransaction& coralTransaction=con.coralTransaction();
  coralTransaction.start(true);
  std::set<std::string> result=coralTransaction.nominalSchema().listTables();
  coralTransaction.commit();
  for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
    std::cout<<"table name "<<*it<<std::endl;
  }
  con.disconnect();
  delete session;
}
