#include "CondCore/DBCommon/interface/GenericRef.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "testCondObj.h"
#include <string>
#include <iostream>
int main(){
  cond::DBSession* session=new cond::DBSession;
  session->configuration().setMessageLevel(cond::Error);
  session->configuration().setAuthenticationMethod(cond::XML);
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  conHandler.registerConnection("sourcedata","sqlite_file:source.db",0);
  conHandler.registerConnection("mycopy","sqlite_file:dest.db",0);
  try{
    session->open();
    conHandler.connect(session);
    cond::Connection* source=conHandler.getConnection("sourcedata");
    testCondObj* myobj=new testCondObj;
    myobj->data.insert(std::make_pair(1,"strangestring1"));
    myobj->data.insert(std::make_pair(100,"strangestring2"));
    cond::PoolTransaction& poolTransaction=source->poolTransaction(false);
    poolTransaction.start();
    cond::TypedRef<testCondObj> myref(poolTransaction,myobj);
    myref.markWrite("mycontainer");
    std::string token=myref.token();
    std::cout<<"token "<<token<<std::endl;
    poolTransaction.commit();
    std::cout<<"committed"<<std::endl;
    poolTransaction.start();
    std::cout<<"started"<<std::endl;
    cond::TypedRef<testCondObj> myinstance(poolTransaction,token);
    std::cout<<"mem pointer "<<myinstance.ptr()<<std::endl;
    std::cout<<"read back 1   "<<myinstance->data[1]<<std::endl;
    std::cout<<"read back 100 "<<myinstance->data[100]<<std::endl;
    poolTransaction.commit();
    //end of prepare data
    
    //start of copying data
    cond::Connection* destcon=conHandler.getConnection("mycopy");
    cond::PoolTransaction& destTransaction=destcon->poolTransaction(false);
    poolTransaction=source->poolTransaction(true);
    poolTransaction.start();
    cond::GenericRef mydata(poolTransaction,token,"testCondObj");
    std::string t=mydata.token();
    std::string n=mydata.className();
    std::string m=mydata.containerName();
    destTransaction.start();
    std::string resultToken=mydata.exportTo(destTransaction);
    destTransaction.commit();
    poolTransaction.commit();
    std::cout<<"reading back with generic ref token "<<t<<'\n';
    std::cout<<"class name "<<n<<'\n';
    std::cout<<"container name "<<m<<std::endl;
    std::cout<<"result token from copy "<<resultToken<<std::endl;
    conHandler.disconnectAll();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
  delete session;
}
