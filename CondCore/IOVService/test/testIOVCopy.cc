#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/test/testPayloadObj.h"
#include <iostream>

int main(){
  edmplugin::PluginManager::Config config;
  try{
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    std::string sourceConnect("sqlite_file:source.db");
    std::string destConnect("sqlite_file:dest.db");
    cond::DbConnection connection;
    //connection.configuration().setMessageLevel(coral::Debug);
    connection.configure();
    cond::DbSession sourcedb = connection.createSession();
    sourcedb.open("sqlite_file:source.db");
    cond::DbSession destdb = connection.createSession();
    destdb.open("sqlite_file:dest.db");
    
    cond::IOVEditor sourceIov(sourcedb);
    cond::IOVEditor destIov(destdb);
    sourcedb.transaction().start(false);
    sourceIov.createIOVContainerIfNecessary();
    sourceIov.create(cond::timestamp,1);
    for(int i=0; i<5; ++i){
      std::cout<<"creating test payload obj"<<i<<std::endl;
      testPayloadObj* myobj=new testPayloadObj;
      for(int j=0; j<10; ++j){
        myobj->data.push_back(i+j);
      }
      
      boost::shared_ptr<testPayloadObj> myobjPtr (myobj );
      std::string tok = sourcedb.storeObject(myobjPtr.get(),"testPayloadObjRcd");
      sourceIov.append(i+10, tok);
    }
    std::string iovtoken=sourceIov.proxy().token();
    std::cout<<"iov token "<<iovtoken<<std::endl;
    sourcedb.transaction().commit();
    std::cout<<"source db created "<<std::endl;
    sourcedb.transaction().start(true);
    std::cout<<"source db started "<<std::endl;
    destdb.transaction().start(false);
    std::cout<<"dest db started "<<std::endl;
    destIov.createIOVContainerIfNecessary();
    destIov.create(sourceIov.proxy().timetype(),sourceIov.proxy().lastTill()); 
    std::cout<<"importing... "<<std::endl;
    destIov.import( sourcedb, iovtoken );
    destdb.transaction().commit();
    std::cout<<"destdb committed"<<std::endl;
    sourcedb.transaction().commit();
    std::cout<<"source db committed"<<std::endl;
    std::cout<<"editor deleted"<<std::endl;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
    return -1;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
    return -1;
  }
  return 0;
}
