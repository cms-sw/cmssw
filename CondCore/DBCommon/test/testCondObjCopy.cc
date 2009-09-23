#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "testCondObj.h"
#include <string>
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection connection;
  connection.configuration().setMessageLevel(coral::Error);
  connection.configure();  
  try{
    cond::DbSession sourceSession = connection.createSession();
    sourceSession.open("sqlite_file:source.db");
    testCondObj* myobj=new testCondObj;
    myobj->data.insert(std::make_pair(1,"strangestring1"));
    myobj->data.insert(std::make_pair(100,"strangestring2"));
    sourceSession.transaction().start(false);
    pool::Ref<testCondObj> myref = sourceSession.storeObject(myobj, "mycontainer");
    std::string token=myref.toString();
    std::cout<<"token "<<token<<std::endl;
    sourceSession.transaction().commit();
    std::cout<<"committed"<<std::endl;
    sourceSession.transaction().start(true);
    std::cout<<"started"<<std::endl;
    pool::Ref<testCondObj> myinstance = sourceSession.getTypedObject<testCondObj>( token );
    std::cout<<"mem pointer "<<myinstance.ptr()<<std::endl;
    std::cout<<"read back 1   "<<myinstance->data[1]<<std::endl;
    std::cout<<"read back 100 "<<myinstance->data[100]<<std::endl;
    sourceSession.transaction().commit();
    //end of prepare data
    
    //start of copying data
    cond::DbSession destSession = connection.createSession();
    destSession.open("sqlite_file:dest.db");
    sourceSession.transaction().start(true);
    destSession.transaction().start(false);
    std::string t=destSession.importObject( destSession, token );
    destSession.transaction().commit();
    sourceSession.transaction().commit();
    std::cout<<"reading back with generic ref token "<<t<<'\n';
    destSession.transaction().start(true);
    pool::Ref<testCondObj> newRef = destSession.getTypedObject<testCondObj>( t );  
    std::cout<<"class name "<<newRef.objectType().Name()<<'\n';
    destSession.transaction().commit();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}
