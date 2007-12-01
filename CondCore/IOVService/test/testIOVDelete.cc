#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Ref.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
#include <iostream>
//#include "CondCore/DBCommon/interface/Ref.h"
int main(){
  try{
    cond::DBSession* session=new cond::DBSession(true);
    session->sessionConfiguration().setMessageLevel(cond::Error);
    session->open();
    cond::PoolStorageManager pooldb("sqlite_file:test.db","file:mycatalog.xml",session);
    pooldb.connect();
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    pooldb.startTransaction(false);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      std::cout<<"myobj "<<myobj<<std::endl;
      for(int j=0; j<10; ++j){
        myobj->data.push_back(i+j);
      }
      cond::Ref<testPayloadObj> myobjRef(pooldb,myobj);
      myobjRef.markWrite("testPayloadObjRcd");
      editor->insert(i+10, myobjRef.token());
    }
    std::string iovtoken=editor->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.commit();
    pooldb.startTransaction(false);
    iovmanager.deleteAll(true);
    pooldb.commit();
    delete editor;
    pooldb.disconnect();
    pooldb.connect();
    //same data, delete by tag this time
    cond::IOVEditor* editorNew=iovmanager.newIOVEditor();
    pooldb.startTransaction(false);
    for(int i=0; i<9; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* cid=new testPayloadObj;
      std::cout<<"cid "<<cid<<std::endl;
      for(int j=0; j<15; ++j){
        cid->data.push_back(i+j);
      }
      cond::Ref<testPayloadObj> cidRef(pooldb,cid);
      cidRef.markWrite("testPayloadObjRcd");
      std::cout<<"token"<<cidRef.token()<<std::endl;
      editorNew->insert(i+10, cidRef.token());
    }
    std::cout<<"end of loop1"<<std::endl;
    iovtoken=editorNew->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.commit();
    delete editorNew;
    pooldb.disconnect();
    pooldb.connect();
    pooldb.startTransaction(false);
    cond::IOVEditor* editorNewNew=iovmanager.newIOVEditor();
    for(int i=0; i<10; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* abc=new testPayloadObj;
      for(int j=0; j<7; ++j){
        abc->data.push_back(i+j);
      }
      cond::Ref<testPayloadObj> abcRef(pooldb,abc);
      abcRef.markWrite("testPayloadObjRcd");
      editorNewNew->insert(i+10, abcRef.token());
    }
    iovtoken=editorNewNew->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.commit();
    pooldb.startTransaction(false);
    //editorNewNew->deleteEntries(true);
    pooldb.commit();
    delete editorNewNew;
    pooldb.disconnect();
    session->close();
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
