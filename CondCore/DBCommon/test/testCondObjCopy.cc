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
    boost::shared_ptr<testCondObj> myobj( new testCondObj );
    myobj->data.insert(std::make_pair(1,"strangestring1"));
    myobj->data.insert(std::make_pair(100,"strangestring2"));
    sourceSession.transaction().start(false);
    sourceSession.createDatabase();
    std::string token = sourceSession.storeObject(myobj.get(), "mycontainer");
    std::cout<<"token "<<token<<std::endl;
    sourceSession.transaction().commit();
    std::cout<<"committed"<<std::endl;
    sourceSession.transaction().start(true);
    std::cout<<"started"<<std::endl;
    boost::shared_ptr<testCondObj> myinstance = sourceSession.getTypedObject<testCondObj>( token );
    std::cout<<"mem pointer "<<myinstance.get()<<std::endl;
    std::cout<<"read back 1   "<<myinstance->data[1]<<std::endl;
    std::cout<<"read back 100 "<<myinstance->data[100]<<std::endl;
    sourceSession.transaction().commit();
    //end of prepare data
    
    //start of copying data
    cond::DbSession destSession = connection.createSession();
    destSession.open("sqlite_file:dest.db");
    sourceSession.transaction().start(true);
    destSession.transaction().start(false);
    destSession.createDatabase();
    std::cout << "importing\n";
    std::string t=destSession.importObject( sourceSession, token );
    destSession.transaction().commit();
    sourceSession.transaction().commit();
    std::cout<<"committed"<<std::endl;
    std::cout<<"reading back with generic ref token "<<t<<'\n';
    destSession.transaction().start(true);
    boost::shared_ptr<testCondObj> newRef = destSession.getTypedObject<testCondObj>( t );  
    destSession.transaction().commit();
    std::cout<<"committed"<<std::endl;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}
