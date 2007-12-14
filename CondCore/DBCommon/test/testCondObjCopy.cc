#include "CondCore/DBCommon/interface/Ref.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "testCondObj.h"
//#include <exception>
#include <string>
#include <iostream>
int main(){
  cond::DBSession* session=new cond::DBSession(true);
  session->sessionConfiguration().setMessageLevel(cond::Error);
  session->sessionConfiguration().setAuthenticationMethod(cond::XML);
  try{
    session->open();
    cond::PoolStorageManager sourcedb("sqlite_file:source.db",
				      "file:sourcecatalog.xml",
				      session);
    sourcedb.connect();
    testCondObj* myobj=new testCondObj;
    myobj->data.insert(std::make_pair(1,"strangestring1"));
    myobj->data.insert(std::make_pair(100,"strangestring2"));
    cond::Ref<testCondObj> myref(sourcedb,myobj);
    sourcedb.startTransaction(false);
    myref.markWrite("mycontainer");
    std::string token=myref.token();
    std::cout<<"token "<<token<<std::endl;
    sourcedb.commit();
    sourcedb.startTransaction(true);
    cond::Ref<testCondObj> myinstance(sourcedb,token);
    std::cout<<"mem pointer "<<myinstance.ptr()<<std::endl;
    std::cout<<"read back 1   "<<myinstance->data[1]<<std::endl;
    std::cout<<"read back 100 "<<myinstance->data[100]<<std::endl;
    sourcedb.commit();
    sourcedb.disconnect();
    cond::PoolStorageManager destdb("sqlite_file:dest.db",
				    "file:destcatalog.xml",
				    session);
    destdb.connect();
    destdb.startTransaction(false);
    cond::Ref<testCondObj> newref(destdb,myinstance.ptr());
    newref.markWrite("mycontainer");
    std::cout<<"new token "<<newref.token()<<std::endl;
    destdb.commit();
    destdb.disconnect();
    session->close();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
  delete session;
}
