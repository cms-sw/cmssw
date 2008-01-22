#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"

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
  cond::DBSession* session=new cond::DBSession;
  session->configuration().setMessageLevel( cond::Error );
  session->configuration().setAuthenticationMethod(cond::XML);
  session->configuration().connectionConfiguration()->disablePoolAutomaticCleanUp();
  session->configuration().connectionConfiguration()->setConnectionTimeOut(0);
  session->configuration().connectionConfiguration()->setIdleConnectionCleanupPeriod(10);
  cond::Connection con("sqlite_file:mydata.db",-1);
  session->open();
  con.connect(session);
  testCondObj* myobj=new testCondObj;
  myobj->data.insert(std::make_pair<unsigned int,std::string>(10,"ten"));
  myobj->data.insert(std::make_pair<unsigned int,std::string>(2,"two"));
  cond::PoolTransaction& poolTransaction=con.poolTransaction();
  poolTransaction.start(false);
  std::cout<<"waiting for 20 sec in pool transaction..."<<std::endl;
  wait(20);
  cond::TypedRef<testCondObj> myref(poolTransaction,myobj);
  myref.markWrite("testCondObjContainer");
  poolTransaction.commit();
  cond::CoralTransaction& coralTransaction=con.coralTransaction();
  coralTransaction.start(true);
  std::cout<<"waiting for 20 sec in coral transaction..."<<std::endl;
  wait(20);
  std::set<std::string> result=coralTransaction.nominalSchema().listTables();
  coralTransaction.commit();
  for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
    std::cout<<"table name "<<*it<<std::endl;
  }
  con.disconnect();
  delete session;
}
