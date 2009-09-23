#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
#include <iostream>
//#include "CondCore/DBCommon/interface/Ref.h"
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  try{
    cond::DbConnection connection;
    connection.configuration().setPoolAutomaticCleanUp( false );
    connection.configure();
    cond::DbSession pooldb = connection.createSession();
    pooldb.open("sqlite_file:mytest.db"); 
    cond::IOVService iovmanager(pooldb);
    pooldb.transaction().start(false);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    editor->create(cond::timestamp,1);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      std::cout<<"myobj "<<myobj<<std::endl;
      for(int j=0; j<10; ++j){
        myobj->data.push_back(i+j);
      }
      pool::Ref<testPayloadObj> myobjRef = pooldb.storeObject(myobj,"testPayloadObj");
      editor->append(i+10, myobjRef.toString());
    }
    std::string iovtoken=editor->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    iovmanager.deleteAll(true);
    pooldb.transaction().commit();

    delete editor;
    pooldb.transaction().start(false);
    //same data, delete by tag this time
    cond::IOVEditor* editorNew=iovmanager.newIOVEditor();
    editorNew->create(cond::timestamp,1);
    for(int i=0; i<9; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* cid=new testPayloadObj;
      std::cout<<"cid "<<cid<<std::endl;
      for(int j=0; j<15; ++j){
        cid->data.push_back(i+j);
      }
      pool::Ref<testPayloadObj> cidRef = pooldb.storeObject(cid,"testPayloadObj");
      std::cout<<"token"<<cidRef.toString()<<std::endl;
      editorNew->append(i+10, cidRef.toString());
    }
    std::cout<<"end of loop1"<<std::endl;

    iovtoken=editorNew->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.transaction().commit();
    delete editorNew;

    pooldb.transaction().start(false);
    cond::IOVEditor* editorNewNew=iovmanager.newIOVEditor();
    editorNewNew->create(cond::timestamp, 1);
    for(int i=0; i<10; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* abc=new testPayloadObj;
      for(int j=0; j<7; ++j){
        abc->data.push_back(i+j);
      }
      pool::Ref<testPayloadObj> abcRef = pooldb.storeObject(abc,"testPayloadObj");
      editorNewNew->append(i+10, abcRef.toString());
    }
    iovtoken=editorNewNew->token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.transaction().commit();
    //pooldb.transaction().start();
    //editorNewNew->deleteEntries(true);
    //pooldb.transaction().commit();
    delete editorNewNew;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
