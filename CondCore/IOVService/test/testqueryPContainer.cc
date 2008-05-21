#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    //session->configuration().setMessageLevel(cond::Error);
    //session->configuration().setAuthenticationMethod(cond::XML);
    session->open();
    cond::Connection myconnection("sqlite_file:mytest.db",0); 
    myconnection.connect(session);    
    testPayloadObj* myobj=new testPayloadObj;
    myobj->data.push_back(1);
    myobj->data.push_back(10);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    pooldb.start(false);
    cond::TypedRef<testPayloadObj> myref(pooldb,myobj);
    myref.markWrite("mypayloadcontainer");
    std::string token=myref.token();
    std::cout<<"payload token "<<token<<std::endl;
    cond::IOVService iovmanager(pooldb,cond::timestamp);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    editor->create(1,cond::timestamp);
    editor->insert(20,token);
    std::string iovtok=editor->token();
    std::string cname=iovmanager.payloadContainerName(iovtok);
    pooldb.commit();
    myconnection.disconnect();
    std::cout<<"Payload Container Name: "<<cname<<std::endl;
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
