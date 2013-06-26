#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "testPayloadObj.h"
int main(){
  edmplugin::PluginManager::Config config;
  try{
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    cond::DbConnection connection;
    connection.configuration().setMessageLevel( coral::Debug );
    connection.configure();
    cond::DbSession pooldb = connection.createSession();
    pooldb.open("sqlite_file:mytest.db"); 
    testPayloadObj* myobj=new testPayloadObj;
    myobj->data.push_back(1);
    myobj->data.push_back(10);
    pooldb.transaction().start(false);
    boost::shared_ptr<testPayloadObj> myPtr( myobj );
    std::string token = pooldb.storeObject(myPtr.get(),"mypayloadcontainer");
    std::cout<<"payload token "<<token<<std::endl;
    cond::IOVEditor editor( pooldb );
    std::cout << "creating\n";
    editor.create(cond::timestamp, 2);
    std::cout << "appending";
    editor.append(1,token);
    std::string iovtok=editor.proxy().token();
    std::cout<<"iov token "<<iovtok<<std::endl;
    std::set<std::string> cnames=editor.proxy().payloadClasses();
    pooldb.transaction().commit();
    std::cout<<"Payload Class Names: "<<std::endl;
    for( std::set<std::string>::iterator iC = cnames.begin(); iC != cnames.end(); ++iC ){
      std::cout << *iC <<std::endl;
    }
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
