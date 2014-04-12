#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  try{
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    cond::DbConnection connection;
    connection.configuration().setPoolAutomaticCleanUp( false );
    connection.configure();
    cond::DbSession pooldb = connection.createSession();
    pooldb.open("sqlite_file:mytest.db"); 
    cond::IOVEditor editor( pooldb );
    pooldb.transaction().start(false);
    editor.createIOVContainerIfNecessary();
    editor.create(cond::timestamp,1);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      std::cout<<"myobj "<<myobj<<std::endl;
      for(int j=0; j<10; ++j){
        myobj->data.push_back(i+j);
      }
      boost::shared_ptr<testPayloadObj> myobjPtr( myobj );
      std::string tok = pooldb.storeObject(myobjPtr.get(),"testPayloadObj");
      editor.append(i+10, tok);
    }
    std::string iovtoken=editor.proxy().token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    pooldb.transaction().commit();

    pooldb.transaction().start(false);
    editor.reload();
    editor.deleteEntries(true);
    pooldb.transaction().commit();
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
