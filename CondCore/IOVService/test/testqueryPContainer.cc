#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Ref.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/src/IOV.h"
#include "testPayloadObj.h"
int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->sessionConfiguration().setMessageLevel(cond::Error);
    session->open();
    cond::PoolStorageManager pooldb("sqlite_file:testqueryc.db","file:mycatalog.xml",session);
    pooldb.connect();
    std::cout<<1<<std::endl;
    testPayloadObj* myobj=new testPayloadObj;
    std::cout<<2<<std::endl;
    myobj->data.push_back(1);
    myobj->data.push_back(10);
    std::cout<<3<<std::endl;
    pooldb.startTransaction(false);
    cond::Ref<testPayloadObj> myref(pooldb,myobj);
    std::cout<<4<<std::endl;
    std::cout<<5<<std::endl;
    myref.markWrite("mypayloadcontainer");
    std::cout<<6<<std::endl;
    std::string token=myref.token();
    std::cout<<"payload token "<<token<<std::endl;
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    editor->insert(20,token);
    std::string iovtok=editor->token();
    std::string cname=iovmanager.payloadContainerName(iovtok);
    pooldb.commit();
    pooldb.disconnect();
    std::cout<<"Payload Container Name: "<<cname<<std::endl;
    delete editor;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
