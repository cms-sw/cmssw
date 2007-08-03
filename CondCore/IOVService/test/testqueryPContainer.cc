#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/src/IOV.h"
#include "testPayloadObj.h"
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->configuration().setMessageLevel(cond::Error);
    session->configuration().setAuthenticationMethod(cond::XML);
    static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
    conHandler.registerConnection("mytest","sqlite_file:testqueryc.db","file:mycatalog.xml",0);
    session->open();
    conHandler.connect(session);
    std::cout<<1<<std::endl;
    testPayloadObj* myobj=new testPayloadObj;
    std::cout<<2<<std::endl;
    myobj->data.push_back(1);
    myobj->data.push_back(10);
    std::cout<<3<<std::endl;
    cond::Connection* myconnection=conHandler.getConnection("mytest");    
    cond::PoolTransaction& pooldb=myconnection->poolTransaction(false);
    pooldb.start();
    cond::TypedRef<testPayloadObj> myref(pooldb,myobj);
    std::cout<<4<<std::endl;
    myref.markWrite("mypayloadcontainer");
    std::cout<<6<<std::endl;
    std::string token=myref.token();
    std::cout<<"payload token "<<token<<std::endl;
    cond::IOVService iovmanager(pooldb);
    std::cout<<"a"<<std::endl;
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    std::cout<<"b"<<std::endl;
    editor->insert(20,token);
    std::cout<<"c"<<std::endl;
    std::string iovtok=editor->token();
    std::cout<<"d"<<std::endl;
    std::string cname=iovmanager.payloadContainerName(iovtok);
    pooldb.commit();
    std::cout<<"Payload Container Name: "<<cname<<std::endl;
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
